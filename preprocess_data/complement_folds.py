import json
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import gensim
from collections import defaultdict
from operator import itemgetter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DATA_DIR = os.environ.get("IREV_RAW_DATA_DIR", "/home/eduardo/dataset")
EMB_DIR = os.path.join(REPO_ROOT, "embeddings_reviews")
if REPO_ROOT not in sys.path:
    # append (not insert(0, ...)): this must not shadow preprocess_data's
    # own utils.py/text_preprocess.py with iRev's top-level utils/ package.
    sys.path.append(REPO_ROOT)

# Funções customizadas (Certifique-se de que os arquivos abaixo estão acessíveis)
from add_eduardo.leave_one_out_cv import get_loocv_fold_normalized
from text_preprocess import build_doc, countNum
from utils import get_count, now

# name -> (raw file format, filename under RAW_DATA_DIR)
RAW_FILES = {
    "imdb": ("parquet", "imdb.parquet"),
    "amazon": ("parquet", "amazon.parquet"),
    "tripadvisor": ("parquet", "tripadvisor.parquet"),
    "beauty": ("jsonl", "All_Beauty_reviews.jsonl"),
    "musical_instruments": ("jsonl", "Musical_Instruments_reviews.jsonl"),
}


def load_raw_dataset(dataset_name):
    """Loads and normalizes a raw dataset to columns:
    user_id, item_id, ratings, reviews, timestamp, review_emb.

    amazon/imdb/tripadvisor already ship numeric userId/itemId (parquet).
    beauty/musical_instruments are raw Amazon-2023 jsonl (string user_id /
    parent_asin) and need factorizing to numeric ids.
    """
    kind, filename = RAW_FILES[dataset_name]
    path = os.path.join(RAW_DATA_DIR, filename)

    if kind == "parquet":
        df = pd.read_parquet(path)
        if "user_id" in df.columns:
            # string user_id (e.g. "AGSNCTF...") collides with the numeric
            # userId column we actually want to key on -- drop it.
            df = df.drop(columns="user_id")
        df = df.rename(columns={
            "userId": "user_id", "itemId": "item_id",
            "rating": "ratings", "review": "reviews",
        })
        if "timestamp" not in df.columns:
            # amazon.parquet ships with no timestamp column at all.
            # get_loocv_fold_normalized needs one to order interactions
            # per user -- fall back to row order as a recency surrogate.
            print(f"WARNING: {dataset_name} has no timestamp column; "
                  f"using row order as a chronology surrogate.")
            df["timestamp"] = np.arange(len(df))
    else:  # raw Amazon-2023 jsonl
        df = pd.read_json(path, lines=True)
        df["user_id"] = pd.factorize(df["user_id"])[0]
        df["item_id"] = pd.factorize(df["parent_asin"])[0]
        df = df.rename(columns={"rating": "ratings", "text": "reviews"})

    df = df[["user_id", "item_id", "ratings", "reviews", "timestamp"]].copy()
    df["ratings"] = df["ratings"].astype(float)
    df["reviews"] = df["reviews"].fillna("").astype(str)

    emb_path = os.path.join(EMB_DIR, f"{dataset_name}.npy")
    if os.path.exists(emb_path):
        vectors = np.load(emb_path)
        if len(vectors) == len(df):
            df["review_emb"] = list(vectors)
        else:
            print(f"WARNING: {emb_path} has {len(vectors)} rows but {dataset_name} "
                  f"data has {len(df)} rows -- not aligned, zero-filling instead.")
            df["review_emb"] = list(np.zeros((len(df), 768), dtype=np.float32))
    else:
        df["review_emb"] = list(np.zeros((len(df), 768), dtype=np.float32))

    return df

# === FUNÇÕES DE SUPORTE INTEGRADAS (Para evitar erros de importação) ===

def padding_ids(id_list, max_len, padding_idx):
    """Garante que a lista de IDs de interação tenha tamanho fixo."""
    if len(id_list) >= max_len:
        return id_list[:max_len]
    else:
        return id_list + [padding_idx] * (max_len - len(id_list))

def padding_doc(doc_list, max_doc_len=500):
    """Garante que o documento tenha um tamanho fixo de palavras."""
    padded_docs = []
    for doc in doc_list:
        if len(doc) >= max_doc_len:
            padded_docs.append(doc[:max_doc_len])
        else:
            padded_docs.append(doc + [0] * (max_doc_len - len(doc)))
    return np.array(padded_docs), max_doc_len

def padding_text(text_list, num):
    """Garante que a lista de reviews (já tokenizadas) tenha tamanho fixo."""
    if len(text_list) >= num:
        return text_list[:num]
    padding = [[0] * len(text_list[0]) for _ in range(num - len(text_list))]
    return text_list + padding

def remap_ids(df, u_map, i_map):
    """Mapeia IDs originais para índices sequenciais baseados no treino."""
    df['user_id'] = df['user_id'].map(u_map)
    df['item_id'] = df['item_id'].map(i_map)
    return df.dropna(subset=['user_id', 'item_id']).astype({'user_id': int, 'item_id': int})


def build_letter_embedding(df_ctx, id_col, num_entities, emb_dim, rating_filter=None):
    """Mean-pools 'review_emb' per user/item, optionally restricted by rating."""
    matrix = np.zeros((num_entities, emb_dim), dtype=np.float32)
    if rating_filter is not None:
        df_ctx = df_ctx[rating_filter(df_ctx['ratings'])]
    if df_ctx.empty:
        return matrix

    grouped = df_ctx.groupby(id_col)['review_emb'].apply(lambda x: np.mean(np.vstack(x), axis=0))
    for idx, emb in grouped.items():
        matrix[idx] = emb.astype(np.float32)
    return matrix

# pos/neg pivot as a fraction of the max rating: 3 on a 1-5 scale, 6 on 1-10
# (the same 0.6 as new_evaluation.user_thresholds' lower bound)
LETTER_PIVOT_FRACTION = 0.6

# Dense Rating_Matrix.npy (still read by hrdr_ZeroShot) is only written below
# this many cells: for a large/sparse catalog (e.g. Musical Instruments: ~214k
# users x ~123k items) it would be >100GB per fold. LETTER reads the sparse
# Rating_Matrix_sparse.npz instead, which is always written.
MAX_DENSE_MATRIX_CELLS = 500_000_000


def save_letter_files(train_path, data_train, data_val, user_count, item_count, emb_dim):
    """Writes everything models/letter.py loads for one fold."""
    ratings = data_train.drop_duplicates(['user_id', 'item_id'], keep='last')
    rating_matrix = sp.csr_matrix(
        (ratings['ratings'].to_numpy(np.float32),
         (ratings['user_id'].to_numpy(), ratings['item_id'].to_numpy())),
        shape=(user_count, item_count))
    sp.save_npz(f"{train_path}/Rating_Matrix_sparse.npz", rating_matrix)

    large = user_count * item_count > MAX_DENSE_MATRIX_CELLS
    if not large:
        np.save(f"{train_path}/Rating_Matrix.npy", rating_matrix.toarray())
    # Embedding tables scale linearly (users/items x emb_dim), so they are
    # always written; float16 on large catalogs halves disk (~3GB -> ~1.5GB
    # per fold), and models/letter.py casts back to float32 on load.
    emb_dtype = np.float16 if large else np.float32

    # train reviews only: pooling val reviews too puts each val pair's own review (and,
    # through the pos/neg split, its rating bucket) into the features it is scored with
    letter_ctx = data_train
    pivot = LETTER_PIVOT_FRACTION * letter_ctx['ratings'].max()
    for side, id_col, n in (('user', 'user_id', user_count), ('item', 'item_id', item_count)):
        for suffix, rating_filter in (('', None),
                                      ('_pos', lambda r: r > pivot),
                                      ('_neg', lambda r: r <= pivot)):
            emb = build_letter_embedding(letter_ctx, id_col, n, emb_dim, rating_filter)
            np.save(f"{train_path}/letter_{side}_review_emb{suffix}.npy", emb.astype(emb_dtype))

# ===========================================================================


def load_word2vec():
    w2v_path = os.path.join(REPO_ROOT, "language_models", "GoogleNews-vectors-negative300.bin")
    try:
        return gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    except Exception:
        print("Aviso: Word2Vec não carregado. Usando inicialização aleatória.")
        return {}


def write_split(fold_folder, common_path, data_train, data_val, data_test, pre_word2v, global_stats):
    """Writes one train/val/test split (raw ids; target 'z_score' if present, else 'ratings') in the layout config.py reads.

    global_stats is filled on the first call (vocabulary, w2v, max lengths, saved in common_path)
    and reused by later calls, so every split shares the same word space.
    """
    train_path = f"{fold_folder}/train"
    for p in [f'{fold_folder}/train', f'{fold_folder}/val', f'{fold_folder}/test']:
        os.makedirs(p, exist_ok=True)

    # 2. Mapeamento de IDs (Consistente para este Fold)
    u_map = {uid: i for i, uid in enumerate(get_count(data_train, 'user_id'))}
    i_map = {iid: i for i, iid in enumerate(get_count(data_train, 'item_id'))}

    data_train = remap_ids(data_train, u_map, i_map)
    data_val = remap_ids(data_val, u_map, i_map)
    data_test = remap_ids(data_test, u_map, i_map)

    # +2: índices de padding usados em user_item2id / item_user2id
    user_count_target = len(u_map) + 2
    item_count_target = len(i_map) + 2

    # índice interno -> id original (para parear resultados com outros projetos)
    np.save(f"{fold_folder}/user_ids.npy", np.array(list(u_map)))
    np.save(f"{fold_folder}/item_ids.npy", np.array(list(i_map)))

    # 3. Dicionários de Interação (Necessário para user_item2id de cada fold)
    u_iid_dict = defaultdict(list)
    i_uid_dict = defaultdict(list)
    for row in data_train.itertuples():
        u_iid_dict[row.user_id].append(row.item_id)
        i_uid_dict[row.item_id].append(row.user_id)

    # 4. Extração de pares ID-Score
    def extract_data(df_split):
        x = df_split[['user_id', 'item_id']].values
        score_col = 'z_score' if 'z_score' in df_split.columns else 'ratings'
        y = df_split[score_col].values
        return x, y

    x_train, y_train = extract_data(data_train)
    x_val, y_val = extract_data(data_val)
    x_test, y_test = extract_data(data_test)

    # 5. Salvar arquivos dinâmicos (Folder específico da Fold)
    np.save(f"{fold_folder}/val/Val.npy", x_val)
    np.save(f"{fold_folder}/val/Val_Score.npy", y_val)
    np.save(f"{fold_folder}/test/Test.npy", x_test)
    np.save(f"{fold_folder}/test/Test_Score.npy", y_test)
    np.save(f"{train_path}/Train.npy", x_train)
    np.save(f"{train_path}/Train_Score.npy", y_train)

    # Notas originais (o Score acima pode ser o z-score): a avaliação de ranking define
    # 'liked' sobre a escala original (new_evaluation.user_thresholds).
    np.save(f"{fold_folder}/val/Val_Rating.npy", data_val['ratings'].values)
    np.save(f"{fold_folder}/test/Test_Rating.npy", data_test['ratings'].values)
    np.save(f"{train_path}/Train_Rating.npy", data_train['ratings'].values)

    # 5b. LETTER: matriz de ratings do treino + embeddings semânticos (geral/pos/neg)
    # Recalculado a cada fold pois u_map/i_map mudam de fold para fold.
    save_letter_files(train_path, data_train, data_val,
                      user_count_target, item_count_target, data_train['review_emb'].iloc[0].shape[0])

    # 6. Dicionários de Texto (por Fold -- necessários para tokenização abaixo)
    u_rev_dict, i_rev_dict = defaultdict(list), defaultdict(list)
    u_emb_dict, i_emb_dict = defaultdict(list), defaultdict(list)

    for row in data_train.itertuples():
        rev = str(row.reviews).strip() or "<unk>"
        u_rev_dict[row.user_id].append(rev)
        i_rev_dict[row.item_id].append(rev)
        u_emb_dict[row.user_id].append(row.review_emb)
        i_emb_dict[row.item_id].append(row.review_emb)

    # 7. Vocabulário, Word2Vec e Docs BERT -- computados 1x na Fold 0 e
    #    compartilhados (via common/) por todas as folds, para manter o
    #    mesmo espaço de vocabulário/word-embeddings entre folds.
    if not global_stats:
        print("Processando vocabulário e arquivos pesados na pasta 'common'...")
        vocab, u_doc_words, i_doc_words, _, _, topics = build_doc(u_rev_dict, i_rev_dict)
        word_index = {w: i for i, w in enumerate(vocab.keys(), 1)}
        word_index['<unk>'] = 0

        _, _, _, _, _, u_pReviewLen, u_pSentLen = countNum(u_rev_dict)
        _, _, _, _, _, i_pReviewLen, i_pSentLen = countNum(i_rev_dict)
        maxSentLen = max(u_pSentLen, i_pSentLen)

        global_stats.update({
            "word_index": word_index, "u_max_r": u_pReviewLen,
            "i_max_r": i_pReviewLen, "r_max_len": maxSentLen
        })

        # Word2Vec
        vocab_list = sorted(word_index.items(), key=itemgetter(1))
        # seeded: grid_search.py hashes w2v.npy into study names, so an unseeded rebuild of
        # the same split would orphan every finished Optuna study
        rng = np.random.default_rng(0)
        w2v = [pre_word2v[w] if w in pre_word2v else rng.uniform(-1.0, 1.0, (300,)) for w, _ in vocab_list]
        np.save(f"{common_path}/w2v.npy", np.array(w2v))

        # NOTE: userFineReviews/itemFineReviews/userFineDoc/itemFineDoc
        # (BERT-doc embeddings from build_bert_doc) are deliberately
        # NOT generated here. config.py's _load_npy_files() never
        # reads user_review_fine_path/item_review_fine_path/
        # user_fine_doc_path/item_fine_doc_path, and no model in
        # models/*.py references them either -- dead weight that,
        # with review_emb zero-filled this run (no real embeddings
        # computed -- see chat), was costing 6-8GB PER DATASET of
        # pure zeros for nothing. Re-add only alongside wiring an
        # actual consumer for them.

        if topics is not None: np.save(f"{common_path}/Topic_Matrix.npy", topics)

    # 8. Tokenização de Reviews e Docs (word-index arrays) -- POR FOLD,
    #    usando o vocabulário compartilhado (global_stats, fixado na Fold 0).
    #    Sem isso, unpack_input() em main.py não tem o que carregar --
    #    era o passo que faltava nesta versão do script.
    word_index = global_stats["word_index"]
    u_pReviewLen = global_stats["u_max_r"]
    i_pReviewLen = global_stats["i_max_r"]
    maxSentLen = global_stats["r_max_len"]

    def tokenize_side(rev_dict, n_entities, max_rev_len):
        review2index, doc2index = [], []
        for i in range(n_entities):
            texts = rev_dict.get(i, ["<unk>"])
            review_list = []
            for text in texts:
                tokens = [word_index.get(w, 0) for w in text.split()]
                review_list.append(tokens[:maxSentLen] + [0] * (maxSentLen - len(tokens)))
            review2index.append(padding_text(review_list, max_rev_len))

            doc_words = ' '.join(texts).split()
            doc2index.append([word_index.get(w, 0) for w in doc_words])
        return review2index, doc2index

    userReview2Index, userDoc2Index = tokenize_side(u_rev_dict, len(u_map), u_pReviewLen)
    itemReview2Index, itemDoc2Index = tokenize_side(i_rev_dict, len(i_map), i_pReviewLen)

    userDoc2Index, _ = padding_doc(userDoc2Index)
    itemDoc2Index, _ = padding_doc(itemDoc2Index)

    # int32 (not the numpy default int64): halves disk cost and
    # vocab_size (~50k) fits comfortably. config.py recasts dtype on
    # load anyway (_load_npy_files), so this is disk-only.
    np.save(f"{train_path}/userReview2Index.npy", np.array(userReview2Index, dtype=np.int32))
    np.save(f"{train_path}/itemReview2Index.npy", np.array(itemReview2Index, dtype=np.int32))
    np.save(f"{train_path}/userDoc2Index.npy", userDoc2Index.astype(np.int32))
    np.save(f"{train_path}/itemDoc2Index.npy", itemDoc2Index.astype(np.int32))

    # 9. user_item2id / item_user2id -- POR FOLD (o mapeamento de
    #    interações muda a cada fold, precisa ser gerado sempre, não
    #    só na Fold 0).
    u_pad_idx = item_count_target - 1
    i_pad_idx = user_count_target - 1

    user_item2id = []
    for i in range(user_count_target):
        u_iids = u_iid_dict.get(i, [u_pad_idx])
        user_item2id.append(padding_ids(u_iids, u_pReviewLen, u_pad_idx))

    item_user2id = []
    for i in range(item_count_target):
        i_uids = i_uid_dict.get(i, [i_pad_idx])
        item_user2id.append(padding_ids(i_uids, i_pReviewLen, i_pad_idx))

    np.save(f"{train_path}/user_item2id.npy", np.array(user_item2id, dtype=np.int32))
    np.save(f"{train_path}/item_user2id.npy", np.array(item_user2id, dtype=np.int32))

    # 10. Criar Links Simbólicos (Economia de HD) -- apenas para os
    #     arquivos realmente compartilhados/idênticos entre folds.
    shared_files = ["w2v.npy", "Topic_Matrix.npy"]
    for sf in shared_files:
        target = os.path.abspath(f"{common_path}/{sf}")
        link = f"{train_path}/{sf}"
        if not os.path.exists(link) and os.path.exists(target):
            os.symlink(target, link)

    # 11. Config Stats por Fold
    config_stats = {
        "vocab_size": len(global_stats["word_index"]),
        "user_num": user_count_target,
        "item_num": item_count_target,
        "r_max_len": int(global_stats["r_max_len"]),
        "u_max_r": int(global_stats["u_max_r"]),
        "i_max_r": int(global_stats["i_max_r"])
    }
    with open(f"{fold_folder}/config_stats.json", "w") as f:
        json.dump(config_stats, f, indent=4)


if __name__ == "__main__":
    start_time = time.time()
    datasets = os.environ.get("IREV_DATASETS", "imdb,amazon,beauty,musical_instruments,tripadvisor").split(",")
    dataset_root = os.path.join(REPO_ROOT, "dataset")
    pre_word2v = load_word2vec()

    for dataset_name in datasets:
        print(f"\n{'=' * 40}\nDATASET: {dataset_name}\n{'=' * 40}")

        # Pasta comum para os arquivos de ~52GB (Reviews e Embeddings)
        common_path = os.path.join(dataset_root, f"{dataset_name}_data", "common")
        os.makedirs(common_path, exist_ok=True)

        df_full = load_raw_dataset(dataset_name)

        # Variáveis para armazenar estatísticas globais (geradas na Fold 0)
        global_stats = {}

        # --- LOOP DE FOLDS (0 a 4) ---
        for fold_idx in range(5):
            print(f"--- Gerando Fold {fold_idx} ---")
            fold_folder = os.path.join(dataset_root, f"{dataset_name}_data", f"fold_{fold_idx}")

            # 1. Split LOOCV específico do fold
            data_train, data_val, data_test = get_loocv_fold_normalized(df_full, k_fold_index=fold_idx)
            write_split(fold_folder, common_path, data_train, data_val, data_test, pre_word2v, global_stats)

    print(f"\nFinalizado! Tempo total: {time.time() - start_time:.2f}s")
