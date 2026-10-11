import pandas as pd
import numpy as np
import json
import os

# --- CONFIGURATION ---
# Output layout and file names below match what models/letter.py reads via
# config.py's DefaultConfig.set_path(): fold_{k}/train/letter_*_emb*.npy,
# fold_{k}/train/Rating_Matrix.npy, and fold_{k}/config_stats.json.
datasets = ['amazon','imdb', 'rotten_tomatoes']
PIVOT = 3
EMBEDDING_DIM = 768

# TODO(eduardo): point these at the real source CSV (userId,itemId,rating,review,foldId)
# and matching review-embedding .npy for each dataset.
SOURCE_EMBED_DIR = '../../../../embeddings_reviews'
SOURCE_CSV_DIR = '../../../../dataset'


for DATASET_NAME in datasets:
    BASE_OUTPUT_DIR = f'./dataset/{DATASET_NAME}'
    embed = np.load(f'{SOURCE_EMBED_DIR}/{DATASET_NAME}.npy')
    df = pd.read_csv(f'{SOURCE_CSV_DIR}/{DATASET_NAME}.csv')

    df['embedding'] = [i for i in embed]

    # Important: Mapping must be done on the full dataset so ID X is consistent across folds.
    user_map = {u: i for i, u in enumerate(df['userId'].unique())}
    item_map = {i: x for x, i in enumerate(df['itemId'].unique())}

    df['u_idx'] = df['userId'].map(user_map).astype(int)
    df['i_idx'] = df['itemId'].map(item_map).astype(int)

    num_users = len(user_map)
    num_items = len(item_map)

    print(f"Users: {num_users}, Items: {num_items}")

    # --- Helper Functions ---

    def export_json(sub_df, filepath):
        """Exports interactions to a JSON file: {user_id: [[rating, item_id], ...]}"""
        output = {}
        grouped = sub_df.groupby('u_idx')
        for u_id, group in grouped:
            # List of [rating, item_id]
            interactions = [[float(r), int(i)] for r, i in zip(group['rating'], group['i_idx'])]
            output[str(u_id)] = interactions

        with open(filepath, 'w') as f:
            json.dump(output, f)

    def create_embedding_matrix(df_source, group_col, num_entities, filter_condition=None):
        """Creates an embedding matrix by averaging embeddings of interactions."""
        matrix = np.zeros((num_entities, EMBEDDING_DIM), dtype=np.float32)

        if filter_condition is not None:
            df_filtered = df_source[filter_condition(df_source)]
        else:
            df_filtered = df_source

        if df_filtered.empty:
            return matrix

        # Group and calculate mean of embeddings
        # Note: Requires stacking the list of arrays to compute the mean correctly
        temp_grouped = df_filtered.groupby(group_col)['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0))

        for idx, emb_mean in temp_grouped.items():
            matrix[idx] = emb_mean.astype(np.float32)

        return matrix

    # 3. K-Fold Loop
    folds = sorted(df['foldId'].unique())
    print(f"Starting processing for {len(folds)} folds: {folds}")

    for k in folds:
        print(f"--- Processing FOLD {k} ---")

        # Create specific directory for the fold. LETTER-specific artifacts
        # (letter_*_emb*.npy, Rating_Matrix.npy) live under train/, matching
        # config.py's DefaultConfig.set_path() prefix = f"{data_root}/train".
        FOLD_DIR = os.path.join(BASE_OUTPUT_DIR, f'fold_{k}')
        TRAIN_DIR = os.path.join(FOLD_DIR, 'train')
        os.makedirs(TRAIN_DIR, exist_ok=True)

        # 3.1 Split based on foldId column
        # TEST: Current fold
        test_df = df[df['foldId'] == k].copy()

        # TRAIN_FULL: All other folds
        train_full_df = df[df['foldId'] != k].copy()

        # VAL: Split a portion (e.g., 12.5% of train_full ~ 10% of total) for validation
        val_df = train_full_df.sample(frac=0.125, random_state=42)
        train_df = train_full_df.drop(val_df.index)

        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        # 3.2 Export JSONs
        export_json(train_df, os.path.join(FOLD_DIR, 'train.json'))
        export_json(val_df,   os.path.join(FOLD_DIR, 'val.json'))
        export_json(test_df,  os.path.join(FOLD_DIR, 'test.json'))

        # 4. Generate Embedding Matrices (.npy)
        # METHODOLOGY NOTE: To avoid Data Leakage, we compute embedding means
        # using ONLY training data (train + val).
        context_df = pd.concat([train_df, val_df])

        # A. User Embeddings
        u_all = create_embedding_matrix(context_df, 'u_idx', num_users)
        u_pos = create_embedding_matrix(context_df, 'u_idx', num_users, lambda x: x['rating'] > PIVOT)
        u_neg = create_embedding_matrix(context_df, 'u_idx', num_users, lambda x: x['rating'] <= PIVOT)

        # B. Item Embeddings
        i_all = create_embedding_matrix(context_df, 'i_idx', num_items)
        i_pos = create_embedding_matrix(context_df, 'i_idx', num_items, lambda x: x['rating'] > PIVOT)
        i_neg = create_embedding_matrix(context_df, 'i_idx', num_items, lambda x: x['rating'] <= PIVOT)

        # 5. Save embedding matrices under the names/paths models/letter.py loads
        # (opt.letter_user_emb_path, opt.letter_item_emb_path, ..._pos/_neg_path).
        np.save(os.path.join(TRAIN_DIR, 'letter_user_review_emb.npy'), u_all)
        np.save(os.path.join(TRAIN_DIR, 'letter_item_review_emb.npy'), i_all)

        np.save(os.path.join(TRAIN_DIR, 'letter_user_review_emb_pos.npy'), u_pos)
        np.save(os.path.join(TRAIN_DIR, 'letter_user_review_emb_neg.npy'), u_neg)

        np.save(os.path.join(TRAIN_DIR, 'letter_item_review_emb_pos.npy'), i_pos)
        np.save(os.path.join(TRAIN_DIR, 'letter_item_review_emb_neg.npy'), i_neg)

        # 6. Rating_Matrix.npy: dense TRAIN-only user x item rating matrix used
        # by models/letter.py to build its KNN similarity graph (opt.ratingMatrix_path).
        rating_matrix = np.zeros((num_users, num_items), dtype=np.float32)
        for u_id, i_id, rating in zip(train_df['u_idx'], train_df['i_idx'], train_df['rating']):
            rating_matrix[u_id, i_id] = rating
        np.save(os.path.join(TRAIN_DIR, 'Rating_Matrix.npy'), rating_matrix)

        # 7. config_stats.json: config.py's set_path() reads user_num/item_num
        # from here, and they must match the embedding matrices' row counts above.
        with open(os.path.join(FOLD_DIR, 'config_stats.json'), 'w') as f:
            json.dump({'user_num': num_users, 'item_num': num_items}, f, indent=4)

    print("Generation of 5 folds completed successfully!")