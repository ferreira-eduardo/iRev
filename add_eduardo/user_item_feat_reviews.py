import os
import numpy as np
import pandas as pd
from surprise import Reader, Dataset, SVD


def build_user_item_features(
        df: pd.DataFrame,
        emb_matrix: np.ndarray,  # shape: (num_interactions, d_mpnet)
        user_col="userId",
        item_col="itemId",
        out_dim=None,  # if not None, project to this dim
        seed=42,
):
    assert len(df) == emb_matrix.shape[0], "df rows must align with embedding rows"

    # Map ids to contiguous indices 0..N-1
    user_ids = df[user_col].astype("category")
    item_ids = df[item_col].astype("category")
    u_idx = user_ids.cat.codes.to_numpy()
    i_idx = item_ids.cat.codes.to_numpy()
    n_users = user_ids.cat.categories.size
    n_items = item_ids.cat.categories.size

    X = emb_matrix.astype(np.float32)

    # Optional: random projection (fast + simple). Replace with PCA if you prefer.
    if out_dim is not None and out_dim != X.shape[1]:
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 1.0 / np.sqrt(X.shape[1]), size=(X.shape[1], out_dim)).astype(np.float32)
        X = X @ W  # (N, out_dim)

    # Mean aggregation per user
    user_sum = np.zeros((n_users, X.shape[1]), dtype=np.float32)
    user_cnt = np.zeros((n_users, 1), dtype=np.float32)
    np.add.at(user_sum, u_idx, X)
    np.add.at(user_cnt, u_idx, 1.0)
    user_features = user_sum / np.clip(user_cnt, 1.0, None)

    # Mean aggregation per item
    item_sum = np.zeros((n_items, X.shape[1]), dtype=np.float32)
    item_cnt = np.zeros((n_items, 1), dtype=np.float32)
    np.add.at(item_sum, i_idx, X)
    np.add.at(item_cnt, i_idx, 1.0)
    item_features = item_sum / np.clip(item_cnt, 1.0, None)

    return user_features, item_features, user_ids.cat.categories, item_ids.cat.categories


def save_features(user_features, item_features, dataset, emb_opt, out_dir="../checkpoints"):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"user_features_pmf_{dataset}_{emb_opt}.npy"), user_features)
    np.save(os.path.join(out_dir, f"item_features_pmf_{dataset}_{emb_opt}.npy"), item_features)


def build_pmf_like_features(
        csv_path: str,
        out_dir: str,
        dataset_name: str,
        emb_opt: str,
        id_emb_size: int,
        rating_scale=(1, 10),
        n_epochs=30,
        lr_all=0.005,
        reg_all=0.02,
        seed=42,
):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    # must be [userId, itemId, rating]
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df[["userId", "itemId", "rating"]], reader)

    trainset = data.build_full_trainset()

    algo = SVD(
        n_factors=id_emb_size,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=seed,
    )
    algo.fit(trainset)

    # algo.pu: (n_users, n_factors), algo.qi: (n_items, n_factors)
    user_features = algo.pu.astype(np.float32)
    item_features = algo.qi.astype(np.float32)

    user_path = os.path.join(out_dir, f"user_features_pmf_{dataset_name}_{emb_opt}.npy")
    item_path = os.path.join(out_dir, f"item_features_pmf_{dataset_name}_{emb_opt}.npy")
    np.save(user_path, user_features)
    np.save(item_path, item_features)

    print("Saved:", user_path, user_features.shape)
    print("Saved:", item_path, item_features.shape)


datasets = ['imdb']
path = "../../../dataset/{}_5fold_stratified.csv"
for dataset in datasets:
    build_pmf_like_features(
        csv_path=path.format(dataset),  # adjust
        out_dir="../checkpoints",
        dataset_name=f"{dataset}_data",  # must match self.opt.dataset
        emb_opt="word2vec",  # must match self.opt.emb_opt
        id_emb_size=32,
        rating_scale=(1, 10) if dataset == 'amazon' else (1, 5)
    )
