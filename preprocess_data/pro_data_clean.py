import json
import os
import time
import numpy as np
import pandas as pd
import gensim
from collections import defaultdict
from operator import itemgetter

# Funções customizadas (certifique-se de que os caminhos estão corretos)
from add_eduardo.leave_one_out_cv import get_loocv_fold_normalized
from text_preprocess import *
from utils import *


def remap_ids(df, u_map, i_map):
    """Remapeia IDs de usuário e item para garantir que sejam sequenciais."""
    df['user_id'] = df['user_id'].map(u_map)
    df['item_id'] = df['item_id'].map(i_map)
    return df.dropna(subset=['user_id', 'item_id']).astype({'user_id': int, 'item_id': int})


if __name__ == "__main__":
    start_time = time.time()
    datasets = ['imdb', 'amazon', 'rotten_tomatoes']
    base_path = '../../..'
    w2v_path = '../language_models/GoogleNews-vectors-negative300.bin'

    # Carregamento do Word2Vec Global
    try:
        pre_word2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    except FileNotFoundError:
        print("AVISO: Word2Vec não encontrado. Usando inicialização aleatória.")
        pre_word2v = {}

    for dataset_name in datasets:
        print(f"\n{'=' * 40}\nDATASET: {dataset_name}\n{'=' * 40}")

        # Configuração de Caminhos
        csv_path = f"{base_path}/dataset/{dataset_name}.csv"
        npy_path = f"{base_path}/embeddings_reviews/{dataset_name}.npy"
        save_folder = f"../dataset/{dataset_name}_data/fold_0"
        train_path = f"{save_folder}/train"

        for p in [f'{save_folder}/train', f'{save_folder}/val', f'{save_folder}/test']:
            os.makedirs(p, exist_ok=True)

        # Carga de Dados
        try:
            df = pd.read_csv(csv_path)
            mpnet_vectors = np.load(npy_path) if os.path.exists(npy_path) else np.zeros((len(df), 768))
        except Exception as e:
            print(f"Erro ao carregar {dataset_name}: {e}")
            continue

        # Preparação do DataFrame
        df['review_emb'] = list(mpnet_vectors)
        if 'user_id' in df.columns: df.drop(columns='user_id', inplace=True)
        df.rename(columns={'userId': 'user_id', 'itemId': 'item_id', 'rating': 'ratings', 'text': 'reviews'},
                  inplace=True)

        # --- PASSO 1: SPLIT LOOCV & REMAP ---
        data_train, data_val, data_test = get_loocv_fold_normalized(df, k_fold_index=0)

        # Mapeamento Sequencial de IDs (Baseado apenas no Treino)
        u_map = {uid: i for i, uid in enumerate(get_count(data_train, 'user_id'))}
        i_map = {iid: i for i, iid in enumerate(get_count(data_train, 'item_id'))}

        data_train = remap_ids(data_train, u_map, i_map)
        data_val = remap_ids(data_val, u_map, i_map)
        data_test = remap_ids(data_test, u_map, i_map)

        userNum = len(u_map)
        itemNum = len(i_map)
        userNum_all, itemNum_all = userNum, itemNum  # Sincronização

        # --- PASSO 2: EXTRAÇÃO DE INTERAÇÕES & MATRIZ ---
        rating_matrix = np.zeros([userNum + 2, itemNum + 2], dtype=np.single)


        def extract_data(df_split, is_train=False):
            x, y = [], []
            score_col = 'z_score' if 'z_score' in df_split.columns else 'ratings'
            for row in df_split.itertuples():
                uid, iid, score = int(row.user_id), int(row.item_id), float(getattr(row, score_col))
                x.append([uid, iid])
                y.append(score)
                if is_train: rating_matrix[uid][iid] = score
            return x, y


        x_train, y_train = extract_data(data_train, is_train=True)
        x_val, y_val = extract_data(data_val)
        x_test, y_test = extract_data(data_test)

        np.save(f"{save_folder}/val/Val.npy", np.array(x_val))
        np.save(f"{save_folder}/test/Test.npy", np.array(x_test))
        np.save(f"{save_folder}/val/Val_Score.npy", np.array(y_val))
        np.save(f"{save_folder}/test/Test_Score.npy", np.array(y_test))
        np.save(f"{train_path}/Train.npy", np.array(x_train))
        np.save(f"{train_path}/Train_Score.npy", np.array(y_train))


        np.save(f"{train_path}/Rating_Matrix.npy", rating_matrix)

        # --- PASSO 3: VOCABULÁRIO E DICIONÁRIOS ---
        print(f"{now()} Step 4: Building dictionaries and vocab...")
        u_rev_dict, i_rev_dict = defaultdict(list), defaultdict(list)
        u_iid_dict, i_uid_dict = defaultdict(list), defaultdict(list)
        u_emb_dict, i_emb_dict = defaultdict(list), defaultdict(list)

        for row in data_train.itertuples():
            rev = str(row.reviews).strip() or "<unk>"
            u_rev_dict[row.user_id].append(rev)
            u_iid_dict[row.user_id].append(row.item_id)
            u_emb_dict[row.user_id].append(row.review_emb)

            i_rev_dict[row.item_id].append(rev)
            i_uid_dict[row.item_id].append(row.user_id)
            i_emb_dict[row.item_id].append(row.review_emb)

        # Geração de documentos de texto puro
        vocab, u_doc_words, i_doc_words, _, _, topics = build_doc(u_rev_dict, i_rev_dict)

        word_index = {w: i for i, w in enumerate(vocab.keys(), 1)}
        word_index['<unk>'] = 0

        # Estatísticas de Padding
        _, _, _, _, _, u_pReviewLen, u_pSentLen = countNum(u_rev_dict)
        _, _, _, _, _, i_pReviewLen, i_pSentLen = countNum(i_rev_dict)
        maxSentLen = max(u_pSentLen, i_pSentLen)

        # Geração de documentos baseados em Embedding (BERT/FineDoc)
        # Note: build_bert_doc retorna vetores, não palavras!
        u_emb_dict_p, i_emb_dict_p, u_emb_doc, i_emb_doc = build_bert_doc(u_emb_dict, i_emb_dict, u_pReviewLen,
                                                                          i_pReviewLen)

        # --- PASSO 4: PADDING & INDEXAÇÃO ---

        # Processamento de Usuários
        userReview2Index, userDoc2Index, user_iid_list = [], [], []
        for i in range(userNum):
            # IDs de Interação
            u_iids = u_iid_dict.get(i, [itemNum + 1])
            user_iid_list.append(padding_ids(u_iids, u_pReviewLen, itemNum + 1))

            # Documento (Texto -> Índices)
            d_words = u_doc_words[i] if i < len(u_doc_words) else ["<unk>"]
            userDoc2Index.append([word_index.get(w, 0) for w in d_words])

            # Reviews (Sentenças -> Índices)
            u_reviewList = []
            for text in u_rev_dict.get(i, ["<unk>"]):
                tokens = [word_index.get(w, 0) for w in text.split()]
                u_reviewList.append(tokens[:maxSentLen] + [0] * (maxSentLen - len(tokens)))
            userReview2Index.append(padding_text(u_reviewList, u_pReviewLen))

        # Processamento de Itens
        itemReview2Index, itemDoc2Index, item_uid_list = [], [], []
        for i in range(itemNum):
            # IDs de Interação
            i_uids = i_uid_dict.get(i, [userNum + 1])
            item_uid_list.append(padding_ids(i_uids, i_pReviewLen, userNum + 1))

            # Documento (Texto -> Índices)
            d_words = i_doc_words[i] if i < len(i_doc_words) else ["<unk>"]
            itemDoc2Index.append([word_index.get(w, 0) for w in d_words])

            # Reviews (Sentenças -> Índices)
            i_reviewList = []
            for text in i_rev_dict.get(i, ["<unk>"]):
                tokens = [word_index.get(w, 0) for w in text.split()]
                i_reviewList.append(tokens[:maxSentLen] + [0] * (maxSentLen - len(tokens)))
            itemReview2Index.append(padding_text(i_reviewList, i_pReviewLen))

        # Padding de Documentos (Garantir tamanho fixo doc_len)
        userDoc2Index, _ = padding_doc(userDoc2Index)
        itemDoc2Index, _ = padding_doc(itemDoc2Index)

        # --- PASSO 5: WORD EMBEDDING MAPPING & SALVAMENTO ---
        vocab_list = sorted(word_index.items(), key=itemgetter(1))
        w2v = [pre_word2v[w] if w in pre_word2v else np.random.uniform(-1.0, 1.0, (300,)) for w, _ in vocab_list]

        # Salvamento de arquivos conforme DefaultConfig
        np.save(f"{train_path}/w2v.npy", np.array(w2v))

        # Reviews e IDs
        np.save(f"{train_path}/userReview2Index.npy", np.array(userReview2Index))
        np.save(f"{train_path}/user_item2id.npy", np.array(user_iid_list))
        np.save(f"{train_path}/itemReview2Index.npy", np.array(itemReview2Index))
        np.save(f"{train_path}/item_user2id.npy", np.array(item_uid_list))

        # Documentos (Texto puro - Índices)
        np.save(f"{train_path}/userDoc2Index.npy", np.array(userDoc2Index))
        np.save(f"{train_path}/itemDoc2Index.npy", np.array(itemDoc2Index))

        # Documentos (Fine - Vetores BERT/MPNet)
        np.save(f"{train_path}/userFineReviews.npy", u_emb_dict_p)
        np.save(f"{train_path}/itemFineReviews.npy", i_emb_dict_p)
        np.save(f"{train_path}/userFineDoc.npy", u_emb_doc)
        np.save(f"{train_path}/itemFineDoc.npy", i_emb_doc)

        if topics is not None: np.save(f"{save_folder}/Topic_Matrix.npy", topics)

        # Config Stats
        config_stats = {
            "dataset": dataset_name,
            "vocab_size": len(word_index),
            "r_max_len": int(maxSentLen),
            "u_max_r": int(u_pReviewLen),
            "i_max_r": int(i_pReviewLen),
            "user_num": int(userNum_all + 2),
            "item_num": int(itemNum_all + 2),
            "train_data_size": len(x_train)
        }
        with open(f"{save_folder}/config_stats.json", "w") as f:
            json.dump(config_stats, f, indent=4)

        print(f"{now()} Dataset {dataset_name} concluído com sucesso!")

    print(f"\nTempo total: {time.time() - start_time:.2f}s")