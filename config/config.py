# -*- coding: utf-8 -*-
import numpy as np
import os
import json


class DefaultConfig:
    model = ""
    dataset = ""
    folds_train = [0]
    folds = [0]
    fold = None

    # === BASE CONFIGURATION === #

    # PARALLELISM
    use_gpu = True
    gpu_id = 0

    multi_gpu = False
    gpu_ids = []
    seed = 2023
    num_workers = 4

    # TRAIN LOOP CONFIG
    num_epochs = 5
    batch_size = 64
    print_step = 100

    # OPTIMIZER CONFIG
    optimizer = 'Adam'
    weight_decay = 1e-3
    lr = 1e-3
    loss_method = 'mse'
    drop_out = 0.5

    # WORD AND TEXT EMBEDDINGS CONFIG
    use_word_embedding = True
    emb_opt = 'default'
    doc_len = 500
    word_dim = 300

    # LAYERS CONFIG #
    # --- CONVOLUTION --- #
    filters_num = 100
    kernel_size = 3

    # --- MLP --- #
    fc_dim = 32
    query_mlp_size = 128

    # ------------- #

    # FEATURES CONFIG
    id_emb_size = 32
    num_fea = 1
    use_review = True
    use_doc = True
    self_att = False

    # OUTPUT CONFIG
    r_id_merge = 'cat'
    ui_merge = 'cat'
    output = 'lfm'
    direct_output = False

    # SAVE CONFIG
    fine_step = False
    pth_path = "checkpoints/MPCN_Video_Ga_data_default.pth"

    # TEST CONFIG
    ranking_metrics = False

    # SEED PROTOCOL (new_evaluation): split fixo, uma execução por seed,
    # seleção da época (e parada antecipada) pelo MSE da validação; o ranking
    # sobre todo o catálogo é registrado por época e avaliado na melhor época
    seeds = [2023]
    patience = 3            # épocas sem melhora no MSE da validação
    eval_users = 64         # usuários por vez no ranking
    eval_pairs = 16384      # pares (usuário, item) por forward no ranking
    eval_every = None       # ranking da validação a cada N épocas; None: 10 para BPR (LightGCN), 1 para MSE
    statistical_test = True
    # grid_search.py: treina e devolve só o MSE da validação (sem ranking, checkpoint
    # nem results/), para os trials não sobrescreverem as execuções reportadas
    tuning = False
    diversity_metrics = False

    # ALGORITHMS PARTICULAR CONFIGURATIONS
    hrdr = False
    topics = False
    man = False
    transnet = False

    # LETTER (rating-graph + semantic review embeddings)
    letter_hidden_dim = 128
    letter_edge_ratio = 50

    # LightGCN (BPR; defaults from the paper: d=64, K=3, decay=1e-4, batch 2048)
    lgcn_dim = 64
    lgcn_layers = 3
    lgcn_decay = 1e-4

    # KANN (knowledge-graph entity embeddings + self-/cross-attention)
    kann_num_layers = 2
    kann_d_model = 50
    kann_num_heads = 4
    kann_dff = 1024
    kann_freeze_entity_emb = True

    # APH (aspect hypergraph + FM; d1 = d2 = 8..16 in the paper, authors' code: 16)
    aph_dim = 16
    aph_top_ratio = 0.9     # fraction of a node's hyperedges max-pooled in the aspect fusion
    # Train pairs drop the hyperedges of their own review (its sentiment gives the target away;
    # val/test pairs never have them). False = pre-2026-10-10 behaviour.
    aph_exclude_target = True

    # MAGCL (review-similarity graph contrastive learning; released code: d=64, K=3, 4 negatives)
    magcl_dim = 64
    magcl_layers = 3
    magcl_decay = 8e-5      # the released regularizer_rate (declared there but never applied)
    magcl_ssl_weight = 0.05

    # RLMRec-Con on LightGCN (LLM profiles; released lightgcn_plus.yml, Amazon settings)
    rlmrec_dim = 32
    rlmrec_layers = 3
    rlmrec_keep_rate = 0.8
    rlmrec_reg_weight = 1e-7
    rlmrec_kd_weight = 1e-2
    rlmrec_kd_temperature = 0.2

    # BERT EMBEDDING
    bert = ""

    def set_path(self, name):
        self.data_root = f"./dataset/{name}/fold_{self.fold_idx}"
        prefix = f"{self.data_root}/train"

        # --- DEFINE FEATURES PATH (Isso DEVE vir antes de ler o JSON) ---
        self.user_review_fine_path = f"{prefix}/userFineReviews.npy"
        self.item_review_fine_path = f"{prefix}/itemFineReviews.npy"
        self.item_fine_doc_path = f"{prefix}/itemFineDoc.npy"
        self.user_fine_doc_path = f"{prefix}/userFineDoc.npy"

        self.user_list_path = f'{prefix}/userReview2Index.npy'
        self.user2itemid_path = f'{prefix}/user_item2id.npy'
        self.user_doc_path = f'{prefix}/userDoc2Index.npy'

        self.item_list_path = f'{prefix}/itemReview2Index.npy'
        self.item2userid_path = f'{prefix}/item_user2id.npy'
        self.item_doc_path = f'{prefix}/itemDoc2Index.npy'

        self.topicMatrix_path = f"{self.data_root}/Topic_Matrix.npy"
        self.ratingMatrix_path = f"{prefix}/Rating_Matrix.npy"
        self.letter_rating_matrix_path = f"{prefix}/Rating_Matrix_sparse.npz"
        self.w2v_path = f'{prefix}/w2v.npy'

        # --- LETTER (semantic review embeddings, split by rating pivot) ---
        self.letter_user_emb_path = f'{prefix}/letter_user_review_emb.npy'
        self.letter_item_emb_path = f'{prefix}/letter_item_review_emb.npy'
        self.letter_user_emb_pos_path = f'{prefix}/letter_user_review_emb_pos.npy'
        self.letter_user_emb_neg_path = f'{prefix}/letter_user_review_emb_neg.npy'
        self.letter_item_emb_pos_path = f'{prefix}/letter_item_review_emb_pos.npy'
        self.letter_item_emb_neg_path = f'{prefix}/letter_item_review_emb_neg.npy'

        # --- KANN (review-entity knowledge-graph embeddings) ---
        self.kann_re_emb_path = f'{prefix}/kann_entity_re_emb.npy'
        self.kann_ori_emb_path = f'{prefix}/kann_entity_ori_emb.npy'
        self.kann_user_entities_path = f'{prefix}/kann_user_entities.npy'
        self.kann_item_entities_path = f'{prefix}/kann_item_entities.npy'

        # --- APH (aspect hypergraph, preprocess_data/build_aph_fixed_split.py) ---
        self.aph_user_edges_path = f'{prefix}/aph_user_edges.npy'
        self.aph_item_edges_path = f'{prefix}/aph_item_edges.npy'
        self.aph_stats_path = f'{prefix}/aph_stats.json'

        # --- MAGCL (review-similarity graphs, preprocess_data/build_magcl_fixed_split.py) ---
        self.magcl_user_graph_path = f'{prefix}/magcl_user_graph.npz'
        self.magcl_item_graph_path = f'{prefix}/magcl_item_graph.npz'

        # --- RLMRec (LLM profiles, preprocess_data/{generate_rlmrec_profiles,build_rlmrec_fixed_split}.py) ---
        self.rlmrec_user_prf_path = f'{prefix}/rlmrec_user_prf_emb.npy'
        self.rlmrec_item_prf_path = f'{prefix}/rlmrec_item_prf_emb.npy'

        # --- CARREGA O JSON COM SEGURANÇA ---
        stats_path = f"{self.data_root}/config_stats.json"
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                self.vocab_size = stats.get('vocab_size', self.vocab_size)
                self.r_max_len = stats.get('r_max_len', self.r_max_len)
                self.u_max_r = stats.get('u_max_r', self.u_max_r)
                self.i_max_r = stats.get('i_max_r', self.i_max_r)
                self.user_num = stats.get('user_num', self.user_num)
                self.item_num = stats.get('item_num', self.item_num)

    def parse(self, kwargs):
        self.num_folds = kwargs.pop('num_folds', 5)
        self.fold_idx = kwargs.pop('fold_idx', 0)

        self.set_path(self.setting_path)

        stats_path = f"{self.data_root}/config_stats.json"
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                for k, v in stats.items():
                    if hasattr(self, k):
                        setattr(self, k, v)

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception(f'Config has no key: {k}')
            setattr(self, k, v)

        self._load_npy_files()

    def _load_npy_files(self):
        if self.bert == "finetunning":
            pass
        else:
            self.users_review_list = np.array(np.load(self.user_list_path, allow_pickle=True).tolist(), dtype=np.int64)
            self.items_review_list = np.array(np.load(self.item_list_path, allow_pickle=True).tolist(), dtype=np.int64)

            self.user_doc = np.array(np.load(self.user_doc_path, allow_pickle=True).tolist(), dtype=np.float32)
            self.item_doc = np.array(np.load(self.item_doc_path, allow_pickle=True).tolist(), dtype=np.float32)

        self.user2itemid_list = np.array(np.load(self.user2itemid_path, allow_pickle=True).tolist(), dtype=np.int64)
        self.item2userid_list = np.array(np.load(self.item2userid_path, allow_pickle=True).tolist(), dtype=np.int64)

    def print_config(self):
        print('*************************************************')
        print('ACTUAL USER CONFIG (After Bash & JSON):')
        for k, v in vars(self).items():
            if not k.startswith('__') and not isinstance(v, (np.ndarray, dict)):
                print(f"{k} => {v}")
        print('*************************************************')


class Amazon_data_Config(DefaultConfig):
    setting_path = 'amazon_data'

    vocab_size = 50002
    r_max_len = 55
    u_max_r = 20
    i_max_r = 64

    train_data_size = 1163844
    test_data_size = 145462
    val_data_size = 145462

    user_num = 76066
    item_num = 27357


class Rotten_Tomatoes_data_Config(DefaultConfig):
    setting_path = 'rotten_tomatoes_data'

    vocab_size = 50002
    r_max_len = 30
    u_max_r = 30
    i_max_r = 57

    train_data_size = 433756
    test_data_size = 54070
    val_data_size = 54070

    user_num = 2613
    item_num = 17462


class Beauty_data_Config(DefaultConfig):
    setting_path = 'beauty_data'

    # Placeholders -- overwritten at parse() time by fold_X/config_stats.json,
    # written by preprocess_data/complement_folds.py.
    vocab_size = 50002
    r_max_len = 55
    u_max_r = 20
    i_max_r = 64
    user_num = 1
    item_num = 1


class All_Beauty_data_Config(DefaultConfig):
    # Split fixo do ghc2f (preprocess_data/build_fixed_split.py), sem folds.
    setting_path = 'all_beauty_data'

    # Placeholders -- overwritten at parse() time by fold_0/config_stats.json.
    vocab_size = 50002
    r_max_len = 55
    u_max_r = 20
    i_max_r = 64
    user_num = 1
    item_num = 1


class Musical_Instruments_data_Config(DefaultConfig):
    setting_path = 'musical_instruments_data'

    vocab_size = 50002
    r_max_len = 55
    u_max_r = 20
    i_max_r = 64
    user_num = 1
    item_num = 1


class TripAdvisor_data_Config(DefaultConfig):
    setting_path = 'tripadvisor_data'

    vocab_size = 50002
    r_max_len = 55
    u_max_r = 20
    i_max_r = 64
    user_num = 1
    item_num = 1


class Imdb_data_Config(DefaultConfig):
    setting_path = 'imdb_data'

    vocab_size = 50002
    r_max_len = 138
    u_max_r = 43
    i_max_r = 303

    train_data_size = 210871
    val_data_size = 26359
    test_data_size = 26359

    user_num = 7147
    item_num = 1152
