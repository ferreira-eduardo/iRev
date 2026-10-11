# -*- encoding: utf-8 -*-
"""Treino e avaliação no protocolo de seeds.

Split fixo (fold_0 do dataset; sem LOOCV por folds, os dados são esparsos demais),
uma execução por seed para medir a variação do modelo. A época é escolhida pelo
nDCG@10 da validação ranqueando todo o catálogo do treino, e o melhor checkpoint
recebe o relatório completo de new_evaluation na validação e no teste.

    python main.py train --dataset=All_Beauty_data --model=MPCN --seeds=[1,2,3]
    python main.py test  --dataset=All_Beauty_data --model=MPCN --seeds=[1,2,3]
    python main.py summary --dataset=All_Beauty_data
"""
import gc
import os
import random
from contextlib import contextmanager

import fire
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# IMPORTAÇÕES DO SEU PROJETO
import config
import models
from dataset import ReviewData
from framework import Model
from new_evaluation.new_evaluation import SELECTION, Protocol, pair_scorer, save_report, summarize
from utils.utils import *

# torch.set_float32_matmul_precision('high')
#
# torch.backends.cudnn.benchmark = True

RESULTS_DIR = "results/new_protocol"
CHECKPOINT_DIR = "checkpoints/seeds"
REPORTED = ("ndcg@10", "recall@10", "coverage@10", "aplt@10", "novelty@10")


def device_arrays(opt):
    """Tabelas de reviews, docs e vizinhos no device, carregadas uma vez por opt.

    O ranking sobre todo o catálogo faz milhões de forwards; indexar no device
    evita copiar as mesmas linhas da CPU a cada lote.
    """
    if getattr(opt, "_device_arrays", None) is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tables = {
            "users_review_list": opt.users_review_list,
            "items_review_list": opt.items_review_list,
            "user2itemid_list": opt.user2itemid_list,
            "item2userid_list": opt.item2userid_list,
            "user_doc": opt.user_doc,
            "item_doc": opt.item_doc,
        }
        opt._device_arrays = {k: torch.from_numpy(v.astype(np.int64)).to(device) for k, v in tables.items()}
    return opt._device_arrays


def unpack_input(opt, x):
    """
    Refatorado para ser compatível com processamento em lote e diversos tipos de entrada.
    """
    if isinstance(x, (list, tuple)):
        # Se for uma lista de pares [(u1, i1), ...] vinda do zip ou DataLoader
        x = np.array(x)

    t = device_arrays(opt)
    device = t["users_review_list"].device
    uids = torch.from_numpy(x[:, 0].astype(np.int64)).to(device)
    iids = torch.from_numpy(x[:, 1].astype(np.int64)).to(device)

    return [
        t["users_review_list"][uids],
        t["items_review_list"][iids],
        uids,
        iids,
        t["user2itemid_list"][uids],
        t["item2userid_list"][iids],
        t["user_doc"][uids],
        t["item_doc"][iids],
    ]


def model_output(model, opt, pairs):
    """Nota prevista para cada par (usuário, item), como no treino."""
    output = model(unpack_input(opt, pairs))
    # Se o modelo retorna tupla (como ALFM), pegamos apenas a predição
    if isinstance(output, tuple):
        output = output[0]
    return output.reshape(-1)


@contextmanager
def eval_rng(seed):
    """RNG próprio para a avaliação: modelos com amostragem no forward (o gumbel do MPCN
    continua amostrando em eval) dão o mesmo ranking para a mesma seed, e o fluxo
    aleatório do treino não é afetado."""
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_opt(kwargs):
    if 'dataset' not in kwargs:
        raise Exception("Dataset not provided.")
    opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(dict(kwargs, fold_idx=0))  # split fixo: sempre a fold_0 do dataset
    if not isinstance(opt.seeds, (list, tuple)):
        opt.seeds = [opt.seeds]
    return opt


def build_model(opt):
    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
    return model


def checkpoint_path(opt, seed):
    return f"{CHECKPOINT_DIR}/{opt.model}_{opt.dataset}_seed{seed}_{opt.emb_opt}.pth"


def results_path(opt, name):
    return f"{RESULTS_DIR}/{opt.dataset}/{name}.csv"


# @PREDICT: REGRESSÃO (MSE)
@torch.no_grad()
def regression_mse(model, data_loader, opt):
    model.eval()
    total_mse = 0.0
    for pairs, scores in tqdm(data_loader, desc="Regressão", leave=False):
        scores = torch.as_tensor(scores, dtype=torch.float32, device=device_arrays(opt)["user_doc"].device)
        total_mse += F.mse_loss(model_output(model, opt, pairs), scores, reduction='sum').item()
    return total_mse / len(data_loader.dataset)


def batched(fn, opt, n):
    """torch.cat de fn(a, b) sobre [0, n) em fatias de até opt.eval_batch.

    A memória por par varia muito entre modelos (co-atenção palavra a palavra no MPCN,
    por exemplo), então a fatia cai pela metade a cada falta de memória e fica assim
    para o resto da execução.
    """
    opt.eval_batch = getattr(opt, "eval_batch", opt.eval_pairs)
    while True:
        try:
            return torch.cat([fn(a, min(a + opt.eval_batch, n)) for a in range(0, n, opt.eval_batch)])
        except torch.OutOfMemoryError:
            if opt.eval_batch == 1:
                raise
            torch.cuda.empty_cache()
            opt.eval_batch //= 2
            print(f"[OOM] reduzindo o lote de avaliação para {opt.eval_batch} pares")


def pairwise_scorer(model, opt, items):
    """Um forward completo por par (usuário, item): serve para qualquer modelo."""
    return pair_scorer(lambda pairs: batched(lambda a, b: model_output(model, opt, pairs[a:b]), opt, len(pairs)),
                       items, opt.eval_pairs)


def tower_scorer(model, opt, items):
    """Para redes separáveis (lado do usuário só depende do usuário, idem para o item).

    As features de todos os itens saem uma vez e as de cada usuário uma vez por lote;
    por par só rodam a fusão e a predição. Só vale para os pesos atuais.
    """
    catalog = np.asarray(items, dtype=np.int64)
    device = device_arrays(opt)["user_doc"].device
    catalog_t = torch.from_numpy(catalog).to(device)

    def net_features(users, its):
        return model.net(unpack_input(opt, np.stack([users, its], axis=1)))

    # cada item pareado com um usuário qualquer: numa rede separável o lado do item não muda
    item_fea = batched(lambda a, b: net_features(np.zeros(b - a, dtype=np.int64), catalog[a:b])[1],
                       opt, len(catalog))

    def scores(users):
        users = np.asarray(users, dtype=np.int64)
        user_fea = batched(lambda a, b: net_features(users[a:b], np.resize(catalog, b - a))[0], opt, len(users))
        users_t = torch.from_numpy(users).to(device)
        n = len(catalog)

        def block(a, b):
            k = torch.arange(a, b, device=device)
            u, i = k // n, k % n
            return model.predict_from_features(user_fea[u], item_fea[i], users_t[u], catalog_t[i]).reshape(-1)

        return batched(block, opt, len(users) * n).reshape(len(users), n)

    return scores


def factorized_scorer(model, opt, items):
    """Para redes com user_side/item_side/score_matrix (LETTER): sem cabeça de fusão,
    o score é um produto interno, então o catálogo é codificado uma vez e cada lote
    de usuários vira um matmul. Só vale para os pesos atuais."""
    net = model.net
    device = device_arrays(opt)["user_doc"].device
    catalog_t = torch.from_numpy(np.asarray(items, dtype=np.int64)).to(device)
    item_side = batched(lambda a, b: net.item_side(catalog_t[a:b]), opt, len(catalog_t))

    def scores(users):
        users_t = torch.from_numpy(np.asarray(users, dtype=np.int64)).to(device)
        user_side = batched(lambda a, b: net.user_side(users_t[a:b]), opt, len(users_t))
        return net.score_matrix(user_side, item_side)

    return scores


@torch.no_grad()
def model_scorer(model, opt, protocol):
    """score_fn(users) -> (b, N) sobre o catálogo do protocolo.

    Usa tower_scorer quando ele reproduz o forward par a par numa amostra (redes
    separáveis, sem amostragem no forward); senão, pairwise_scorer.
    """
    model.eval()
    slow = pairwise_scorer(model, opt, protocol.items)
    if hasattr(model.net, "score_matrix"):
        fast = factorized_scorer(model, opt, protocol.items)
        users = protocol.rows["val"]["user"].unique()[:2].tolist()
        sample = pairwise_scorer(model, opt, protocol.items[:256])(users)
        if torch.allclose(fast(users)[:, :256], sample, rtol=1e-4, atol=1e-4):
            return fast
        print("[EVAL] factorized_scorer diverge do forward par a par; usando pairwise_scorer")
    if not model.has_fusion_head():
        return slow
    try:
        fast = tower_scorer(model, opt, protocol.items)
        users = protocol.rows["val"]["user"].unique()[:2].tolist()
        sample = pairwise_scorer(model, opt, protocol.items[:256])(users)
        if torch.allclose(fast(users)[:, :256], sample, rtol=1e-4, atol=1e-5):
            return fast
    except Exception as e:  # saídas da rede fora do formato (B, num_fea, D)
        print(f"[EVAL] tower_scorer indisponível ({type(e).__name__}: {e})")
    return slow


def evaluate(score_fn, opt, protocol, name, info):
    """Relatório completo na validação e no teste, salvo em results/new_protocol/<dataset>/<name>.csv."""
    with eval_rng(info["seed"]):
        reports = {split: protocol.report(score_fn, split, opt.eval_users) for split in ("val", "test")}
    save_report(results_path(opt, name), {"dataset": opt.dataset, **info}, reports)
    for split, metrics in reports.items():
        print(f"  {split}: " + " | ".join(f"{k}={metrics[k]:.4f}" for k in REPORTED))
    return reports


def evaluate_popularity(opt, protocol):
    print("[REFERÊNCIA] Popularidade no treino")
    evaluate(protocol.popularity_scorer(), opt, protocol, "Popularity",
             {"model": "Popularity", "seed": 0, "epoch": None})


# @TEST FUNCTION
def test(**kwargs):
    opt = load_opt(kwargs)
    protocol = Protocol(opt.data_root)
    print(protocol.describe())
    torch.set_grad_enabled(False)

    for seed in opt.seeds:
        path = checkpoint_path(opt, seed)
        print(f"\n{'=' * 50}")
        print(f"TESTANDO: {opt.model} | DATASET: {opt.dataset} | SEED: {seed}")
        print(f"CHECKPOINT: {path}")
        print(f"{'=' * 50}")

        if not os.path.exists(path):
            print(f"[SKIP] Arquivo não encontrado: {path}")
            continue

        # mantém a melhor época registrada no treino, se houver
        name = f"{opt.model}_seed{seed}"
        epoch = pd.read_csv(results_path(opt, name))["epoch"].iloc[0] if os.path.exists(results_path(opt, name)) else None

        model = build_model(opt)
        model.load(path)
        evaluate(model_scorer(model, opt, protocol), opt, protocol, name,
                 {"model": opt.model, "seed": seed, "epoch": epoch})

        del model
        torch.cuda.empty_cache()
        gc.collect()

    summary(dataset=opt.dataset)


def summary(dataset, split="test"):
    """Média ± desvio entre seeds por modelo (results/new_protocol/<dataset>/summary.csv)."""
    table = summarize(f"{RESULTS_DIR}/{dataset}", split)
    cols = ["seeds"] + [f"{m}_{s}" for m in REPORTED for s in ("mean", "std") if f"{m}_{s}" in table]
    print(f"\n[{split}] média e desvio entre seeds:\n{table[cols].round(4).to_string()}")


def model_objective(opt):
    """'bpr' para modelos de feedback implícito (LightGCN), 'mse' para os de nota."""
    return getattr(getattr(models, opt.model), "objective", "mse")


def negative_sampler(train_pairs, opt):
    """Negativos uniformes entre os itens do treino, fora do histórico de treino do usuário
    (UniformSample do LightGCN). Colisões são resorteadas algumas vezes."""
    device = device_arrays(opt)["user_doc"].device
    pairs = torch.from_numpy(np.asarray(train_pairs, dtype=np.int64)).to(device)
    items = torch.unique(pairs[:, 1])
    seen = torch.unique(pairs[:, 0] * opt.item_num + pairs[:, 1])

    def sample(users):
        neg = items[torch.randint(len(items), users.shape, device=device)]
        for _ in range(10):
            hit = torch.isin(users * opt.item_num + neg, seen)
            if not hit.any():
                break
            neg[hit] = items[torch.randint(len(items), (int(hit.sum()),), device=device)]
        return neg

    return sample


# @TRAIN LOOP
def train_seed(kwargs, seed, protocol, train_data, val_data):
    """Treina uma seed. Modelos de nota (MSE): a época (e a parada antecipada) é escolhida pelo
    menor MSE da validação; o nDCG@10 só é registrado. Modelos BPR (LightGCN): treinados com
    BPR e escolhidos pelo nDCG@10 da validação, como no artigo. Devolve o critério a minimizar
    da melhor época (MSE, ou -nDCG@10 para BPR)."""
    opt = load_opt(kwargs)
    set_seed(seed)

    print(f"\n{'#' * 40}\nSEED {seed}\n{'#' * 40}")
    print(f"Training: {opt.model} | DATASET: {opt.dataset} | SEED: {seed}")

    model = build_model(opt)

    # model = torch.compile(model)
    # torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=opt.num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    bpr = model_objective(opt) == "bpr"
    # BPR: a regularização é o L2 dos embeddings dentro de bpr_loss (lgcn_decay), não o weight_decay do Adam
    weight_decay = 0 if bpr else opt.weight_decay
    params = model.parameters()
    if hasattr(model.net, "no_decay_parameters"):
        # vieses por usuário/item fora do weight_decay: o L2 acoplado do Adam encolhe todas as
        # linhas a cada passo, e quem tem 1-2 avaliações perde o próprio viés (IMDB, só vieses:
        # MSE de validação 6.06 com weight_decay=1e-3 contra 5.37 sem)
        no_decay = {id(p) for p in model.net.no_decay_parameters()}
        params = list(params)
        params = [{"params": [p for p in params if id(p) not in no_decay]},
                  {"params": [p for p in params if id(p) in no_decay], "weight_decay": 0.0}]
    optimizer = optim.Adam(params, lr=opt.lr, weight_decay=weight_decay)
    if bpr:
        sample_negatives = negative_sampler(train_data.data, opt)
    # tuning (grid_search.py): só o MSE da validação importa; nada de checkpoint nem ranking,
    # para não sobrescrever os resultados e checkpoints das execuções reportadas
    path = None if opt.tuning else checkpoint_path(opt, seed)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    best_epoch, best_crit, waited = -1, float("inf"), 0
    crit_name = f"Val {SELECTION}" if bpr else "Val MSE"
    # Ranking da validação a cada eval_every épocas (o LightGCN original avalia a cada 10);
    # para BPR a paciência conta avaliações, não épocas
    eval_every = opt.eval_every or (10 if bpr else 1)
    for epoch in range(opt.num_epochs):
        model.train()
        total_train_loss = 0

        for batch_data, scores in tqdm(train_loader, desc=f"Seed {seed} - Epoch {epoch}"):
            # Converte scores para Float e move para GPU
            scores = torch.FloatTensor(scores).cuda() if opt.use_gpu else torch.FloatTensor(scores)

            optimizer.zero_grad()
            if bpr:
                _, _, users, pos = unpack_input(opt, batch_data)[:4]
                # num_negatives > 1 (MAGCL): neg vira (B, k); LightGCN segue com um por positivo
                k = getattr(model.net, "num_negatives", 1)
                neg = sample_negatives(users.unsqueeze(1).expand(-1, k) if k > 1 else users)
                loss = model.net.bpr_loss(users, pos, neg)
            else:
                output = model_output(model, opt, batch_data)
                loss = F.mse_loss(output, scores, reduction='mean')
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        log = f"Epoch {epoch} Finalizada. Train Loss: {total_train_loss / len(train_loader):.4f}"
        rank_epoch = (epoch + 1) % eval_every == 0 or epoch == opt.num_epochs - 1
        if not bpr:
            val_mse = regression_mse(model, val_loader, opt)
            crit = val_mse
            log += f" | Val MSE: {val_mse:.4f}"
        if rank_epoch and (bpr or not opt.tuning):
            with eval_rng(seed):
                val = protocol.accuracy(model_scorer(model, opt, protocol), "val", opt.eval_users)
            if bpr:
                crit = -val[SELECTION]
            log += f" | Val {SELECTION}: {val[SELECTION]:.4f} | Val recall@10: {val['recall@10']:.4f}"
        print(log)
        if bpr and not rank_epoch:
            continue  # BPR só é selecionado (e conta paciência) nas épocas avaliadas

        if crit < best_crit:
            best_crit, best_epoch, waited = crit, epoch, 0
            if path:
                torch.save(model.state_dict(), path)
            print(f"--> Melhoria: {crit_name} {abs(best_crit):.4f}")
        else:
            waited += 1
            if waited >= opt.patience:
                unit = f"avaliações (a cada {eval_every} épocas)" if bpr else "épocas"
                print(f"Parada antecipada: {opt.patience} {unit} sem melhora no {crit_name}")
                break

    if opt.tuning:
        print(f"[SEED {seed}] melhor época {best_epoch}: {crit_name} {abs(best_crit):.4f}")
    else:
        print(f"[SEED {seed}] melhor época {best_epoch}: avaliando no catálogo completo")
        model.load(path)
        evaluate(model_scorer(model, opt, protocol), opt, protocol, f"{opt.model}_seed{seed}",
                 {"model": opt.model, "seed": seed, "epoch": best_epoch})

    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    return best_crit


def train(**kwargs):
    opt = load_opt(kwargs)
    protocol = None
    if not opt.tuning or model_objective(opt) == "bpr":  # BPR escolhe a época pelo ranking da validação
        protocol = Protocol(opt.data_root)
        print(protocol.describe())
    if not opt.tuning and not os.path.exists(results_path(opt, "Popularity")):
        evaluate_popularity(opt, protocol)

    # Mesmos dados para todas as seeds; só a inicialização e a ordem dos lotes mudam
    train_data = ReviewData(opt.data_root, mode="Train")
    val_data = ReviewData(opt.data_root, mode="Val")

    seed_min_losses = [train_seed(kwargs, seed, protocol, train_data, val_data) for seed in opt.seeds]

    print("\nTreinamento de todas as seeds concluído.")
    if not opt.tuning:
        summary(dataset=opt.dataset)

    avg_min_loss = float(np.mean(seed_min_losses))
    print(f"GRID_SEARCH_METRIC={avg_min_loss:.6f}")
    return avg_min_loss


if __name__ == "__main__":
    fire.Fire()
