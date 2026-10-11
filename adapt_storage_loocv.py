import os

dataset_name = "rotten_tomatoes_data"  # mude para o seu
base_path = f"./{dataset_name}"
folds = [0, 1, 2, 3, 4]

# Arquivos que são IDENTICOS em todas as folds
shared_files = [
    "itemFineDoc.npy", "itemFineReviews.npy", "userFineDoc.npy",
    "userFineReviews.npy", "itemReview2Index.npy", "userReview2Index.npy",
    "itemDoc2Index.npy", "userDoc2Index.npy", "w2v.npy", "Rating_Matrix.npy",
    "item_user2id.npy"  # Verifique se este mapeamento é global no seu script
]

# 1. Cria a pasta comum
common_dir = os.path.join(base_path, "common")
os.makedirs(common_dir, exist_ok=True)

# 2. Move os arquivos da Fold 0 para a Common (se ainda estiverem lá)
for f in shared_files:
    src = os.path.join(base_path, "fold_0", "train", f)
    dst = os.path.join(common_dir, f)
    if os.path.exists(src) and not os.path.exists(dst):
        print(f"Movendo {f} para common...")
        os.rename(src, dst)

# 3. Cria links simbólicos nas pastas de todas as folds
for fold in folds:
    fold_train_dir = os.path.join(base_path, f"fold_{fold}", "train")
    os.makedirs(fold_train_dir, exist_ok=True)

    for f in shared_files:
        target = os.path.join(common_dir, f)  # Onde o arquivo real está
        link = os.path.join(fold_train_dir, f)  # Onde o modelo espera encontrar

        if not os.path.exists(link):
            # Cria o link simbólico (equivalente ao ln -s no terminal)
            os.symlink(os.path.abspath(target), link)
            print(f"Link criado: Fold {fold} -> {f}")