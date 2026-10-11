import argparse
import copy
import gc
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# from evaluate import evaluate_ranking

dataset = 'amazon'

args = argparse.ArgumentParser(description='args')
args.add_argument('--dataset_name', default=dataset, type=str)
args.add_argument('--batch', default=256, type=int)
args.add_argument('--dims', default=768, type=int)
args.add_argument('--lr', default=0.0001, type=float)
args.add_argument('--hidden', default=128, type=int)
args.add_argument('--reg_lr', default=1, type=float)
args.add_argument('--CL', default=0, type=int)
args.add_argument('--aa', default=0, type=int)
args.add_argument('--cl_lr', default=1, type=float)
args.add_argument('--aa_lr', default=1, type=float)
args.add_argument('--early_stop', default=6, type=int)
args.add_argument('--dataset', default='MEU_DATA', type=str)
args.add_argument('--pivot', default=3, type=int)
args.add_argument('--edge_ratio', default=50, type=int)
args.add_argument('--device', default=0, type=int)
args.add_argument('--rand', default=10, type=int)
args.add_argument('--epochs', default=5, type=int)
args, _ = args.parse_known_args()

torch.manual_seed(args.rand)
np.random.seed(args.rand)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.device}'
device = "cuda" if torch.cuda.is_available() else "cpu"


class ReviewDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)

        self.data = []
        for user_id, samples in data.items():
            for sample in samples:
                self.data.append((int(user_id), sample[0], sample[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(train_path, val_path, test_path):
    train_data = ReviewDataset(train_path)
    val_data = ReviewDataset(val_path)
    test_data = ReviewDataset(test_path)
    return train_data, val_data, test_data


def collate_fn(batch):
    user_ids = torch.tensor([x[0] for x in batch], dtype=torch.long)
    ratings = torch.tensor([x[1] for x in batch], dtype=torch.float32)
    items = torch.tensor([x[2] for x in batch], dtype=torch.long)

    return user_ids, items, ratings


class Model(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim, CL, aa, reviews, ratings):
        super(Model, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_pos_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_neg_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_pos_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_neg_embedding = nn.Embedding(num_items, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_users = num_users
        self.num_items = num_items

        self.user_embedding.weight = nn.Parameter(torch.from_numpy(reviews[0]))
        self.item_embedding.weight = nn.Parameter(torch.from_numpy(reviews[1]))
        self.user_pos_embedding.weight = nn.Parameter(torch.from_numpy(reviews[2]))
        self.user_neg_embedding.weight = nn.Parameter(torch.from_numpy(reviews[4]))
        self.item_pos_embedding.weight = nn.Parameter(torch.from_numpy(reviews[3]))
        self.item_neg_embedding.weight = nn.Parameter(torch.from_numpy(reviews[5]))

        self.user_embedding.weight.requires_grad = False
        self.user_pos_embedding.weight.requires_grad = False
        self.user_neg_embedding.weight.requires_grad = False
        self.item_embedding.weight.requires_grad = False
        self.item_pos_embedding.weight.requires_grad = False
        self.item_neg_embedding.weight.requires_grad = False

        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.user_p = nn.Embedding(num_users, hidden_dim)
        self.item_p = nn.Embedding(num_items, hidden_dim)
        nn.init.xavier_uniform_(self.user_p.weight)
        nn.init.xavier_uniform_(self.item_p.weight)

        self.uFC = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout()
        )
        self.iFC = nn.Sequential(
            nn.Linear(num_users, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout()
        )
        self.ruFC = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout()
        )
        self.riFC = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout()
        )
        self.rupFC = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout()
        )
        self.runFC = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout()
        )
        self.ripFC = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout()
        )
        self.rinFC = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout()
        )
        self.nFC = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        self.ugnn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ignn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upnn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ipnn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.unnn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.innn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.num_users = num_users
        self.num_items = num_items
        self.CL = CL
        self.aa = aa
        self.ReLU = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.reviews = reviews
        self.ratings = ratings
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        self.user_rating = torch.from_numpy(ratings[0].astype('float32')).to(device)
        self.item_rating = torch.from_numpy(ratings[1].astype('float32')).to(device)

        self.rr_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.3)
        self.rr_attn_2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.3)
        self.rr_attn_up = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.3)
        self.rr_attn_un = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.3)
        self.rr_attn_ip = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.3)
        self.rr_attn_in = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True, dropout=0.3)

    def graph_edge__(self, user_emb, item_emb, user_ids, item_ids):
        k_ratio = args.edge_ratio
        num_u = max(1, int(torch.ceil(torch.tensor(k_ratio / 100.0 * self.num_users))))
        num_i = max(1, int(torch.ceil(torch.tensor(k_ratio / 100.0 * self.num_items))))


        _, u_sims, _ = self.calculate_similarity_in_chunks(user_emb, user_ids, num_u)
        _, i_sims, _ = self.calculate_similarity_in_chunks(item_emb, item_ids, num_i)

        return None, None, u_sims, i_sims

    def mark_unique_elements(self, tensor):
        unique_elements, inverse_indices, counts = torch.unique(tensor, return_inverse=True, return_counts=True)

        boolean_tensor = torch.zeros_like(tensor, dtype=torch.bool, device=device)

        first_occurrence = torch.zeros_like(unique_elements, dtype=torch.bool, device=device)
        boolean_tensor[first_occurrence[inverse_indices].logical_not()] = True
        first_occurrence[inverse_indices] = True

        return boolean_tensor

    def calculate_similarity_in_chunks(self, full_emb_matrix, ids, top_k, chunk_size=50000):
        device = full_emb_matrix.device
        ids = ids.to(device)

        # Normaliza apenas as queries do batch atual (pequeno, não gasta VRAM)
        query_emb = torch.nn.functional.normalize(full_emb_matrix[ids], p=2, dim=1)

        total_elements = full_emb_matrix.size(0)
        all_values = []
        all_indices = []

        with torch.no_grad():
            for i in range(0, total_elements, chunk_size):
                end = min(i + chunk_size, total_elements)

                # 1. Pega o chunk da matriz original
                chunk = full_emb_matrix[i:end]

                # 2. Normaliza apenas este chunk [ChunkSize, Dim]
                norm_chunk = torch.nn.functional.normalize(chunk, p=2, dim=1)

                # 3. Calcula similaridade parcial [Batch, ChunkSize]
                sim_chunk = torch.mm(query_emb, norm_chunk.t())

                # 4. Pega o top_k local e guarda
                k_eff = min(top_k, sim_chunk.size(1))
                vals, inds = torch.topk(sim_chunk, k=k_eff, dim=1)

                all_values.append(vals)
                all_indices.append(inds + i)

                # LIMPEZA CRÍTICA: deleta os tensores pesados do loop imediatamente
                del chunk, norm_chunk, sim_chunk

            # Consolida apenas os resultados do Top-K (muito leves)
            combined_values = torch.cat(all_values, dim=1)
            combined_indices = torch.cat(all_indices, dim=1)

            final_values, final_rel_indices = torch.topk(combined_values, k=top_k, dim=1)
            final_indices = torch.gather(combined_indices, 1, final_rel_indices)

        # Criação da matriz esparsa
        sparse_values = torch.nn.functional.softmax(final_values, dim=1).flatten()
        rows = torch.arange(ids.size(0), device=device).view(-1, 1).repeat(1, top_k).flatten()
        cols = final_indices.flatten()

        adj_sparse = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            sparse_values,
            size=(ids.size(0), total_elements),
            device=device
        ).coalesce()

        return None, adj_sparse, None

    def forward(self, user_ids, item_ids, rating, clip):
        # 1. Obter adjacência esparsa (já otimizado para chunks)
        _, _, u_sims, i_sims = self.graph_edge__(self.user_rating / 5, self.item_rating / 5, user_ids, item_ids)

        device = self.user_bias.weight.device
        u_sims = u_sims.to(device)
        i_sims = i_sims.to(device)

        def get_sparse_projection(sim_matrix, embedding_layer, projection_layer, ids):
            neighbor_indices = sim_matrix.indices()[1].unique()
            all_needed_indices = torch.unique(torch.cat([ids, neighbor_indices]))

            projected_subset = projection_layer(embedding_layer.weight[all_needed_indices])

            full_projected = torch.zeros((embedding_layer.num_embeddings, projected_subset.size(1)), device=device)
            full_projected[all_needed_indices] = projected_subset
            return full_projected

        u_emb_all = get_sparse_projection(u_sims, self.user_embedding, self.ruFC, user_ids)
        up_emb_all = get_sparse_projection(u_sims, self.user_pos_embedding, self.rupFC, user_ids)
        un_emb_all = get_sparse_projection(u_sims, self.user_neg_embedding, self.runFC, user_ids)

        i_emb_all = get_sparse_projection(i_sims, self.item_embedding, self.riFC, item_ids)
        ip_emb_all = get_sparse_projection(i_sims, self.item_pos_embedding, self.ripFC, item_ids)
        in_emb_all = get_sparse_projection(i_sims, self.item_neg_embedding, self.rinFC, item_ids)

        user_r_embeds = self.ugnn1(torch.sparse.mm(u_sims, u_emb_all)) + u_emb_all[user_ids]
        item_r_embeds = self.ignn1(torch.sparse.mm(i_sims, i_emb_all)) + i_emb_all[item_ids]

        user_pos_r_embeds = self.upnn1(torch.sparse.mm(u_sims, up_emb_all)) + up_emb_all[user_ids]
        user_neg_r_embeds = self.unnn1(torch.sparse.mm(u_sims, un_emb_all)) + un_emb_all[user_ids]

        item_pos_r_embeds = self.ipnn1(torch.sparse.mm(i_sims, ip_emb_all)) + ip_emb_all[item_ids]
        item_neg_r_embeds = self.innn1(torch.sparse.mm(i_sims, in_emb_all)) + in_emb_all[item_ids]

        user_biases = self.user_bias(user_ids).squeeze()
        item_biases = self.item_bias(item_ids).squeeze()

        dot_product = (user_r_embeds * item_r_embeds).sum(1)

        dot_product += (user_pos_r_embeds * item_r_embeds).sum(1)
        dot_product -= (user_neg_r_embeds * item_r_embeds).sum(1)

        dot_product += (user_r_embeds * item_pos_r_embeds).sum(1)
        dot_product -= (user_r_embeds * item_neg_r_embeds).sum(1)

        max_rat = 10 if args.dataset_name == 'imdb' else 5
        prediction = dot_product + user_biases + item_biases
        if clip == 1:
            prediction = torch.clamp(prediction, 1, max_rat)

        return prediction, 0, 0, 0


def calculate_mask_with_candidates(norm_matrix, total_elements, ids, top_k):
    device = norm_matrix.device
    ids = ids.to(device)
    query_emb = norm_matrix[ids]  # [Batch, Dim]

    num_queries = ids.size(0)
    chunk_size = 50000
    all_values = []
    all_indices = []

    with torch.no_grad():
        for i in range(0, total_elements, chunk_size):
            end = min(i + chunk_size, total_elements)
            chunk_ref = norm_matrix[i:end].t()

            sim_chunk = torch.mm(query_emb, chunk_ref)

            # Pega o top_k dentro deste chunk
            k_eff = min(top_k, sim_chunk.size(1))
            vals, inds = torch.topk(sim_chunk, k=k_eff, dim=1)

            all_values.append(vals)
            all_indices.append(inds + i)  # Ajusta o índice global

            # Limpa memória temporária do chunk
            del sim_chunk

        # Consolida os resultados dos chunks
        combined_values = torch.cat(all_values, dim=1)
        combined_indices = torch.cat(all_indices, dim=1)

        # Faz o top_k final sobre os melhores de cada chunk
        final_values, final_rel_indices = torch.topk(combined_values, k=top_k, dim=1)
        final_indices = torch.gather(combined_indices, 1, final_rel_indices)

    # Criação da matriz esparsa (na GPU)
    sparse_values = torch.nn.functional.softmax(final_values, dim=1).flatten()
    rows = torch.arange(num_queries, device=device).view(-1, 1).repeat(1, top_k).flatten()
    cols = final_indices.flatten()

    adj_sparse = torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        sparse_values,
        size=(num_queries, total_elements),
        device=device
    ).coalesce()

    return None, adj_sparse, None

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def ratings_to_array(ratings):
    z = []
    for rating in ratings:
        r_array = [0 for i in range(0, 5)]
        r_array[int(rating.item() - 1)] = 1
        z.append(r_array)
    return torch.tensor(z).float()


def train(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    total_mse = 0
    total_cl = 0
    total_aa = 0
    for user_ids, item_ids, ratings in tqdm(dataloader):
        user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)

        optimizer.zero_grad()
        outputs, reg_loss, cl_loss, aa_loss = model(user_ids, item_ids, ratings, clip=0)

        loss_ = criterion(outputs, ratings)

        loss = loss_ + args.reg_lr * reg_loss + args.cl_lr * cl_loss + args.aa_lr * aa_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        total_loss += loss.item()
        total_mse += loss_.item()
        total_cl += args.cl_lr * cl_loss
        total_aa += args.aa_lr * aa_loss
    return total_loss / len(dataloader), total_mse / len(dataloader), total_cl / len(dataloader), total_aa / len(
        dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for user_ids, item_ids, ratings in dataloader:
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)

            outputs, _, _, _ = model(user_ids, item_ids, ratings)
            loss = criterion(outputs, ratings)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_rmse(model, dataloader, device, criterion):
    model.eval()
    predictions = []
    actuals = []
    total_mae = 0
    total_mse = 0
    total_rmse = 0
    with torch.no_grad():
        for user_ids, item_ids, ratings in dataloader:
            user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)

            outputs, _, _, _ = model(user_ids, item_ids, ratings, clip=1)
            mae = (ratings - outputs).abs().mean()
            mse = nn.MSELoss()(outputs, ratings)
            rmse = np.sqrt(mse.item())
            total_mae = total_mae + mae.item()
            total_mse = total_mse + mse.item()
            total_rmse = total_rmse + rmse
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(ratings.cpu().numpy())

    return total_rmse / len(dataloader), total_mse / len(dataloader), total_mae / len(dataloader)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)


def load_ratings(dataset, max_reviewer_index, max_asin_index):
    with open(dataset, 'r') as file:
        data = json.load(file)

    reviewer_ratings = np.zeros((max_reviewer_index, max_asin_index))
    asin_ratings = np.zeros((max_asin_index, max_reviewer_index))

    for reviewer_index, ratings in data.items():
        reviewer_index = int(reviewer_index)
        for rating, asin_index in ratings:
            reviewer_ratings[reviewer_index, asin_index] = rating
            asin_ratings[asin_index, reviewer_index] = rating

    return reviewer_ratings.astype('int'), asin_ratings.astype('int')


def load_json_to_df(filepath):
    """Lê o JSON gerado no passo 1 e converte para DataFrame (userId, itemId, rating)"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    rows = []
    for uid, interactions in data.items():
        for rating, iid in interactions:
            rows.append([int(uid), int(iid), float(rating)])

    return pd.DataFrame(rows, columns=['userId', 'itemId', 'rating'])


# def main():
#     datasets = ['amazon', 'rotten_tomatoes']
#     folds = [0, 1, 2, 3, 4]
#
#     # Configurações de performance do sistema
#     torch.backends.cudnn.benchmark = True
#
#     for dataset in datasets:
#         results_file = f'{dataset}_results_summary.csv'
#         dataset_results = []  # Lista limpa para cada dataset
#
#         for fold in folds:
#             print(f"\n{'=' * 30}\nIniciando Dataset: {dataset} | Fold: {fold}\n{'=' * 30}")
#
#             # --- 1. Definição de Caminhos ---
#             base_fold_path = f'./dataset/{dataset}/fold_{fold}'
#             train_path = os.path.join(base_fold_path, 'train.json')
#             val_path = os.path.join(base_fold_path, 'val.json')
#             test_path = os.path.join(base_fold_path, 'test.json')
#
#             # --- 2. Carga de Dados e Embeddings ---
#             train_data, val_data, test_data = load_data(train_path, val_path, test_path)
#
#             # Carga única de embeddings para evitar I/O redundante
#             def load_npy(name):
#                 return np.load(os.path.join(base_fold_path, name), allow_pickle=True)
#
#             u_review = load_npy('encoded_user_review_v2.npy')
#             i_review = load_npy('encoded_item_review_v2.npy')
#
#             u_p_review = load_npy(f'encoded_user_review_rating_{args.pivot + 1}_or_above_v2.npy')
#             u_n_review = load_npy(f'encoded_user_review_rating_{args.pivot}_or_below_v2.npy')
#             i_p_review = load_npy(f'encoded_item_review_rating_{args.pivot + 1}_or_above_v2.npy')
#             i_n_review = load_npy(f'encoded_item_review_rating_{args.pivot}_or_below_v2.npy')
#
#             num_users, num_items = u_review.shape[0], i_review.shape[0]
#             u_ratings, i_ratings = load_ratings(train_path, num_users, num_items)
#
#             # Dataloaders otimizados (num_workers ideal para L40S é entre 4 e 8)
#             loader_args = {'batch_size': args.batch, 'collate_fn': collate_fn, 'num_workers': 4}
#             train_loader = DataLoader(train_data, shuffle=True, **loader_args)
#             val_loader = DataLoader(val_data, shuffle=False, **loader_args)
#             test_loader = DataLoader(test_data, shuffle=False, **loader_args)
#
#             # --- 3. Inicialização do Modelo ---
#             model = Model(
#                 num_users, num_items,
#                 embedding_dim=args.dims,
#                 hidden_dim=args.hidden,
#                 CL=args.CL, aa=args.aa,
#                 reviews=[u_review, i_review, u_p_review, i_p_review, u_n_review, i_n_review],
#                 ratings=[u_ratings, i_ratings]
#             ).to(device)
#
#             optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
#             criterion = nn.MSELoss()
#
#             # --- 4. Loop de Treinamento com Early Stopping ---
#             best_rmse = float('inf')
#             best_model_state = None
#             patience_counter = 0
#
#             for epoch in range(args.epochs):
#                 model.train()
#                 train_loss, train_mse, cl, aa = train(model, train_loader, optimizer, criterion, device, epoch)
#
#                 model.eval()
#                 val_rmse, val_mse, val_mae = evaluate_rmse(model, val_loader, device, criterion)
#                 print(f'Epoch {epoch + 1:02d} | Val RMSE: {val_rmse:.4f}')
#
#                 if val_rmse < best_rmse:
#                     best_rmse = val_rmse
#                     best_model_state = copy.deepcopy(model.state_dict())
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
#
#                 if patience_counter >= args.early_stop:
#                     print(f"Early stopping na época {epoch + 1}")
#                     break
#
#             # --- 5. Avaliação Final e Ranking ---
#             if best_model_state:
#                 model.load_state_dict(best_model_state)
#
#             test_rmse, test_mse, test_mae = evaluate_rmse(model, test_loader, device, criterion)
#
#             # Ranking (Importante carregar os DFs apenas aqui para economizar RAM)
#             train_df_pd = load_json_to_df(train_path)
#             test_df_pd = load_json_to_df(test_path)
#
#             ranking_results = evaluate_ranking(
#                 model=model, val_df=test_df_pd, train_df=train_df_pd,
#                 all_item_ids=set(range(num_items)), device=device,
#                 threshold=train_df_pd['rating'].max(), k=10
#             )
#
#             print(f"RESULTADO FOLD {fold} -> RMSE: {test_rmse:.4f} | NDCG@10: {ranking_results['ndcg']:.4f}")
#
#             # Consolidar resultados
#             res = {
#                 'fold': fold, 'mae': test_mae, 'rmse': test_rmse, 'mse': test_mse,
#                 **ranking_results  # Desempacota as métricas de ranking diretamente
#             }
#             dataset_results.append(res)
#
#             # --- 6. LIMPEZA CRÍTICA DE MEMÓRIA ---
#             del model, train_loader, val_loader, test_loader, u_review, i_review, u_ratings, i_ratings
#             gc.collect()
#             torch.cuda.empty_cache()
#
#         # Salvar CSV por dataset
#         pd.DataFrame(dataset_results).to_csv(results_file, index=False)
#         print(f"Sucesso: {results_file} gerado.")
#
#
# if __name__ == '__main__':
#     main()
