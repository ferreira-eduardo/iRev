import pandas as pd
import numpy as np
import json
import os


datasets = ['amazon', 'imdb', 'rotten_tomatoes']
PIVOT = 3
EMBEDDING_DIM = 768

# Split config
SPLIT_TYPE = "user"
TEST_FRAC = 0.10
VAL_FRAC = 0.10
RANDOM_STATE = 42



def export_json(sub_df, filepath):
    """Exports interactions to a JSON file: {user_id: [[rating, item_id], ...]}"""
    output = {}
    for u_id, group in sub_df.groupby('u_idx'):
        interactions = [[float(r), int(i)] for r, i in zip(group['rating'], group['i_idx'])]
        output[str(int(u_id))] = interactions

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

    grouped_means = df_filtered.groupby(group_col)['embedding'].apply(
        lambda x: np.mean(np.vstack(x.to_numpy()), axis=0)
    )

    for idx, emb_mean in grouped_means.items():
        matrix[int(idx)] = emb_mean.astype(np.float32)

    return matrix

def split_user_level(df, test_frac, val_frac, seed=42):
    """
    Splits by user so that users do NOT overlap across splits.
    val_frac and test_frac are fractions of the FULL dataset (approx).
    """
    rng = np.random.default_rng(seed)
    users = df['u_idx'].unique()
    rng.shuffle(users)

    n_users = len(users)
    n_test = int(round(test_frac * n_users))
    n_val = int(round(val_frac * n_users))

    test_users = set(users[:n_test])
    val_users  = set(users[n_test:n_test + n_val])
    train_users = set(users[n_test + n_val:])

    train_df = df[df['u_idx'].isin(train_users)].copy()
    val_df   = df[df['u_idx'].isin(val_users)].copy()
    test_df  = df[df['u_idx'].isin(test_users)].copy()

    return train_df, val_df, test_df

def split_interaction_level(df, test_frac, val_frac, seed=42):
    """
    Splits by interactions (rows). Users/items can overlap across splits.
    val_frac and test_frac are fractions of FULL dataset.
    """
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(df)
    n_test = int(round(test_frac * n))
    n_val  = int(round(val_frac * n))

    test_df = df.iloc[:n_test].copy()
    val_df  = df.iloc[n_test:n_test + n_val].copy()
    train_df = df.iloc[n_test + n_val:].copy()

    return train_df, val_df, test_df

for DATASET_NAME in datasets:
    BASE_OUTPUT_DIR = f'./dataset_one/{DATASET_NAME}'
    SPLIT_DIR = os.path.join(BASE_OUTPUT_DIR, 'split')
    os.makedirs(SPLIT_DIR, exist_ok=True)

    embed = np.load(f'../../../../embeddings_reviews/{DATASET_NAME}.npy', allow_pickle=False)
    df = pd.read_csv(f'../../../../dataset/{DATASET_NAME}.csv')

    if len(embed) != len(df):
        raise ValueError(f"[{DATASET_NAME}] embed rows ({len(embed)}) != df rows ({len(df)})")

    # Put embeddings into df
    df['embedding'] = list(embed)

    # IMPORTANT: mapping on full dataset
    user_map = {u: i for i, u in enumerate(df['userId'].unique())}
    item_map = {it: j for j, it in enumerate(df['itemId'].unique())}

    df['u_idx'] = df['userId'].map(user_map).astype(int)
    df['i_idx'] = df['itemId'].map(item_map).astype(int)

    num_users = len(user_map)
    num_items = len(item_map)

    print(f"\n[{DATASET_NAME}] Users: {num_users}, Items: {num_items}, Interactions: {len(df)}")

    # -------- Split (NO KFOLD) --------
    if SPLIT_TYPE.lower() == "user":
        train_df, val_df, test_df = split_user_level(df, TEST_FRAC, VAL_FRAC, seed=RANDOM_STATE)
    elif SPLIT_TYPE.lower() == "interaction":
        train_df, val_df, test_df = split_interaction_level(df, TEST_FRAC, VAL_FRAC, seed=RANDOM_STATE)
    else:
        raise ValueError("SPLIT_TYPE must be 'user' or 'interaction'")

    print(f"[{DATASET_NAME}] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # -------- Export JSONs --------
    export_json(train_df, os.path.join(SPLIT_DIR, 'train.json'))
    export_json(val_df,   os.path.join(SPLIT_DIR, 'val.json'))
    export_json(test_df,  os.path.join(SPLIT_DIR, 'test.json'))

    # -------- Embedding matrices (NO leakage: train+val only) --------
    context_df = pd.concat([train_df, val_df], ignore_index=True)

    # User matrices
    u_all = create_embedding_matrix(context_df, 'u_idx', num_users)
    u_pos = create_embedding_matrix(context_df, 'u_idx', num_users, lambda x: x['rating'] > PIVOT)
    u_neg = create_embedding_matrix(context_df, 'u_idx', num_users, lambda x: x['rating'] <= PIVOT)

    # Item matrices
    i_all = create_embedding_matrix(context_df, 'i_idx', num_items)
    i_pos = create_embedding_matrix(context_df, 'i_idx', num_items, lambda x: x['rating'] > PIVOT)
    i_neg = create_embedding_matrix(context_df, 'i_idx', num_items, lambda x: x['rating'] <= PIVOT)

    # -------- Save .npy files --------
    np.save(os.path.join(SPLIT_DIR, 'encoded_user_review_v2.npy'), u_all)
    np.save(os.path.join(SPLIT_DIR, 'encoded_item_review_v2.npy'), i_all)

    np.save(os.path.join(SPLIT_DIR, f'encoded_user_review_rating_{PIVOT+1}_or_above_v2.npy'), u_pos)
    np.save(os.path.join(SPLIT_DIR, f'encoded_user_review_rating_{PIVOT}_or_below_v2.npy'), u_neg)

    np.save(os.path.join(SPLIT_DIR, f'encoded_item_review_rating_{PIVOT+1}_or_above_v2.npy'), i_pos)
    np.save(os.path.join(SPLIT_DIR, f'encoded_item_review_rating_{PIVOT}_or_below_v2.npy'), i_neg)

print("\nGeneration of Train/Val/Test split completed successfully!")
