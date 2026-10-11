import pandas as pd


def normalize_by_train(train_set: pd.DataFrame, *others: pd.DataFrame):
    """
    Adds per-user 'z_score' and 'is_relevant' columns, with mean/std taken STRICTLY from train.
    Returns (train_set, *others) normalized with the same statistics.
    """
    # We compute mean and std for every user based ONLY on their training history
    user_stats = train_set.groupby('user_id')['ratings'].agg(['mean', 'std']).reset_index()
    user_stats.columns = ['user_id', 'train_mean', 'train_std']

    # Handle cases where std is 0 (users with identical ratings) to avoid div by zero
    user_stats['train_std'] = user_stats['train_std'].replace(0, 1.0).fillna(1.0)

    def normalize(target_df):
        target_df = target_df.merge(user_stats, on='user_id', how='left')
        target_df['z_score'] = (target_df['ratings'] - target_df['train_mean']) / target_df['train_std']
        # Define relevance (e.g., Z-score > 0 means better than their average)
        target_df['is_relevant'] = target_df['z_score'] > 0
        return target_df

    return tuple(normalize(df) for df in (train_set, *others))


def get_loocv_fold_normalized(df: pd.DataFrame, k_fold_index: int):
    """
    Generates Train, Val, and Test sets with Z-score normalized relevance.
    """
    # Temporal Sorting
    df_sorted = df.sort_values(['user_id', 'timestamp']).copy()

    # Filter users with enough history (Min: k + 3)
    min_req = k_fold_index + 3
    user_counts = df_sorted['user_id'].value_counts()
    keep_users = user_counts[user_counts >= min_req].index
    df_filtered = df_sorted[df_sorted['user_id'].isin(keep_users)].copy()

    # Extract Test and Validation (Sliding Window)
    test_set = df_filtered.groupby('user_id').nth(-(k_fold_index + 1)).copy()
    val_set = df_filtered.groupby('user_id').nth(-(k_fold_index + 2)).copy()

    # Extract Training Set
    train_set = df_filtered.groupby('user_id', group_keys=False)[df_filtered.columns].apply(
        lambda x: x.iloc[:-(k_fold_index + 2)]
    ).copy()

    return normalize_by_train(train_set, val_set, test_set)
