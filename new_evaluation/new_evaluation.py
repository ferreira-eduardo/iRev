import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


def rating_levels(train: pd.DataFrame) -> int:
    """Número de níveis da escala de notas (5 para 1-5, 10 para o IMDB 1-10), lido do treino.

    As notas precisam ser inteiras a partir de 1; a perda ordinal usa rating - 1 como índice.
    """
    r = train["rating"]
    if r.min() < 1 or (r % 1 != 0).any():
        raise ValueError(f"notas devem ser inteiras a partir de 1; encontrei min={r.min()}, "
                         f"valores {sorted(r.unique())[:10]}")
    return int(r.max())


def user_thresholds(train: pd.DataFrame, k: float = 5.0,
                    low: float = 0.6, high: float = 0.8) -> tuple[pd.Series, float]:
    """Limiar de positivo por usuário: média encolhida para a global, recortada em [low, high].

    low e high são frações da nota máxima: 0.6 e 0.8 dão [3, 4] na escala 1-5 e
    [6, 8] na escala 1-10. Calculado só com o treino. Devolve os limiares por usuário
    e o limiar global, usado para usuários que não aparecem no treino.
    """
    top = rating_levels(train)
    low, high = low * top, high * top
    global_mean = train["rating"].mean()
    stats = train.groupby("user")["rating"].agg(["sum", "count"])
    shrunk = (stats["sum"] + k * global_mean) / (stats["count"] + k)
    fallback = min(max(global_mean, low), high)
    return shrunk.clip(lower=low, upper=high), fallback


def mark_liked(df: pd.DataFrame, thresholds: pd.Series, fallback: float) -> pd.DataFrame:
    """Acrescenta a coluna 'liked'. Aplicar com os mesmos limiares em treino, validação e teste."""
    limit = df["user"].map(thresholds).fillna(fallback)
    return df.assign(liked=df["rating"] >= limit)


ITEM_CHUNK = 8192            # itens por vez na avaliação (limita a memória da GPU)
KS = (3, 5, 10, 15, 20)      # cortes reportados; o nDCG@10 é o critério de seleção
SELECTION = "ndcg@10"


# ---------- Ranking ----------

@dataclass
class Rankings:
    ranks: np.ndarray    # (n,) posição do alvo; 0 = primeiro lugar
    top: np.ndarray      # (n, max(KS)) índices do catálogo no topo de cada lista
    targets: np.ndarray  # (n,) índice do alvo no catálogo


def evaluation_rows(split: pd.DataFrame, catalog: dict, history: dict) -> pd.DataFrame:
    """Interações avaliáveis: positivas, item conhecido no treino e não repetido do histórico."""
    rows = split[split["liked"] & split["item"].isin(catalog)]
    fresh = [t not in history.get(u, ()) for u, t in zip(rows["user"], rows["item"])]
    return rows[fresh]


@torch.no_grad()
def rank_all(score_fn: Callable[[list], torch.Tensor], rows: pd.DataFrame, catalog: dict,
             history: dict, chunk: int) -> Rankings:
    """Uma passada sobre o catálogo do treino; todas as métricas saem daqui.

    score_fn(users) -> (b, N), na ordem de catalog. O histórico do usuário é excluído.
    Empates contam pela posição média, para não favorecer scores constantes.
    """
    ranks, tops, targets = [], [], []
    for start in range(0, len(rows), chunk):
        part = rows.iloc[start:start + chunk]
        users = part["user"].tolist()
        scores = score_fn(users).float()

        for r, u in enumerate(users):
            seen = [catalog[i] for i in history.get(u, ()) if i in catalog]
            scores[r, seen] = float("-inf")

        target = torch.tensor([catalog[t] for t in part["item"]], device=scores.device)
        t_score = scores.gather(1, target[:, None])
        higher = (scores > t_score).sum(1)
        ties = (scores == t_score).sum(1) - 1

        ranks.append((higher + ties / 2).cpu())
        tops.append(scores.topk(max(KS), dim=1).indices.cpu())
        targets.append(target.cpu())

    return Rankings(torch.cat(ranks).numpy(), torch.cat(tops).numpy(), torch.cat(targets).numpy())


# ---------- Métricas ----------

def accuracy(ranks: np.ndarray, ks: tuple[int, ...] = KS) -> dict[str, float]:
    metrics = {}
    for k in ks:
        metrics[f"ndcg@{k}"] = float(np.where(ranks < k, 1 / np.log2(ranks + 2), 0).mean())
        metrics[f"recall@{k}"] = float((ranks < k).mean())
    return metrics


def head_items(popularity: np.ndarray, share: float = 0.2) -> np.ndarray:
    """Máscara da cabeça: os itens mais populares que somam `share` das interações."""
    order = np.argsort(-popularity)
    cumulative = np.cumsum(popularity[order]) / popularity.sum()
    head = np.zeros(len(popularity), dtype=bool)
    head[order[:np.searchsorted(cumulative, share) + 1]] = True
    return head


def beyond_accuracy(r: Rankings, popularity: np.ndarray, n_users: int, k: int = 10) -> dict[str, float]:
    """Cobertura, concentração e viés de popularidade das listas top-k."""
    top = r.top[:, :k]
    counts = np.bincount(top.ravel(), minlength=len(popularity)).astype(float)
    head = head_items(popularity)

    sorted_counts = np.sort(counts)
    i = np.arange(1, len(counts) + 1)
    gini = ((2 * i - len(counts) - 1) * sorted_counts).sum() / (len(counts) * counts.sum())

    p = counts[counts > 0] / counts.sum()
    entropy = -(p * np.log(p)).sum() / np.log(len(counts))  # normalizada em [0, 1]

    return {
        f"coverage@{k}": float((counts > 0).mean()),
        f"gini@{k}": float(gini),
        f"entropy@{k}": float(entropy),
        f"arp@{k}": float(popularity[top].mean()),
        f"aplt@{k}": float((~head[top]).mean()),
        f"novelty@{k}": float(-np.log2(popularity[top] / n_users).mean()),
    }


def stratified(r: Rankings, groups: np.ndarray, k: int = 10) -> dict[str, float]:
    """nDCG@k por grupo de usuários (ou de alvos)."""
    gains = np.where(r.ranks < k, 1 / np.log2(r.ranks + 2), 0)
    return {f"ndcg@{k}[{g}]": float(gains[groups == g].mean()) for g in np.unique(groups)}


def full_report(r: Rankings, popularity: np.ndarray, n_users: int,
                activity: np.ndarray) -> dict[str, float]:
    target_group = np.where(head_items(popularity)[r.targets], "alvo cabeça", "alvo cauda")
    return {**accuracy(r.ranks), **beyond_accuracy(r, popularity, n_users),
            **stratified(r, activity), **stratified(r, target_group)}


def activity_groups(rows: pd.DataFrame, user_count: dict) -> np.ndarray:
    """Faixas de interações no treino, onde se testa a hipótese do gate."""
    n = rows["user"].map(user_count).fillna(0).to_numpy()
    return np.select([n <= 1, n <= 3], ["1 interação", "2-3 interações"], "4+ interações")


def pair_scorer(forward: Callable[[np.ndarray], torch.Tensor], items: list,
                max_pairs: int) -> Callable[[list], torch.Tensor]:
    """score_fn(users) -> (b, N) contra todo o catálogo, para os modelos do iRev.

    Os modelos do iRev pontuam pares (usuário, item) e vários cruzam os textos dos
    dois lados (co-atenção), então não há um lado do item reaproveitável: cada usuário
    é pontuado contra o catálogo em pedaços de até max_pairs pares.
    forward(pairs) recebe um array (P, 2) de [user, item] e devolve (P,) scores.
    """
    catalog = np.asarray(items, dtype=np.int64)

    def scores(users: list) -> torch.Tensor:
        step = max(1, max_pairs // len(users))
        parts = []
        for j in range(0, len(catalog), step):
            block = catalog[j:j + step]
            pairs = np.stack([np.repeat(users, len(block)), np.tile(block, len(users))], axis=1)
            parts.append(forward(pairs).reshape(len(users), len(block)))
        return torch.cat(parts, dim=1)

    return scores


# ---------- Protocolo: split fixo, seeds, catálogo completo ----------

def read_split(data_root: str, split: str) -> pd.DataFrame:
    """Pares (user, item) e nota original de um split salvo por preprocess_data/complement_folds.py."""
    name = split.capitalize()
    pairs = np.load(f"{data_root}/{split}/{name}.npy")
    rating_path = f"{data_root}/{split}/{name}_Rating.npy"
    if not os.path.exists(rating_path):
        raise FileNotFoundError(f"{rating_path} não existe: gere o dataset com "
                                f"preprocess_data/build_fixed_split.py (salva as notas originais)")
    return pd.DataFrame({"user": pairs[:, 0], "item": pairs[:, 1], "rating": np.load(rating_path)})


class Protocol:
    """Tudo o que a avaliação precisa, calculado só com o treino (e treino + validação no teste).

    - positivo: nota >= limiar do usuário (user_thresholds sobre o treino);
    - catálogo: itens do treino; o alvo precisa estar nele e não ser repetido do histórico;
    - histórico excluído do ranking: treino na validação, treino + validação no teste.
    """

    def __init__(self, data_root: str):
        train, val, test = (read_split(data_root, s) for s in ("train", "val", "test"))
        thresholds, fallback = user_thresholds(train)
        train, val, test = (mark_liked(df, thresholds, fallback) for df in (train, val, test))

        self.items = sorted(train["item"].unique().tolist())
        self.catalog = {item: j for j, item in enumerate(self.items)}
        counts = train["item"].value_counts()
        self.popularity = counts.reindex(self.items).to_numpy(dtype=float)
        self.n_users = train["user"].nunique()
        self.user_count = train["user"].value_counts().to_dict()

        seen_train = train.groupby("user")["item"].agg(set).to_dict()
        seen_trainval = pd.concat([train, val]).groupby("user")["item"].agg(set).to_dict()
        self.history = {"val": seen_train, "test": seen_trainval}
        self.rows = {"val": evaluation_rows(val, self.catalog, seen_train),
                     "test": evaluation_rows(test, self.catalog, seen_trainval)}
        self.sizes = {"val": len(val), "test": len(test)}

    def describe(self) -> str:
        # os splits salvos já não têm itens fora do treino (remap_ids os descarta)
        return " | ".join(f"{s}: {len(r)} avaliáveis de {self.sizes[s]} com item do catálogo"
                          for s, r in self.rows.items()) + f" | catálogo: {len(self.items)} itens"

    def rank(self, score_fn: Callable[[list], torch.Tensor], split: str, chunk: int) -> Rankings:
        return rank_all(score_fn, self.rows[split], self.catalog, self.history[split], chunk)

    def accuracy(self, score_fn: Callable[[list], torch.Tensor], split: str, chunk: int) -> dict[str, float]:
        """Só as métricas de acurácia (seleção da época)."""
        return accuracy(self.rank(score_fn, split, chunk).ranks)

    def report(self, score_fn: Callable[[list], torch.Tensor], split: str, chunk: int) -> dict[str, float]:
        """Relatório completo: acurácia, além da acurácia e estratificado."""
        rankings = self.rank(score_fn, split, chunk)
        activity = activity_groups(self.rows[split], self.user_count)
        return full_report(rankings, self.popularity, self.n_users, activity)

    def popularity_scorer(self) -> Callable[[list], torch.Tensor]:
        """Referência: todos os usuários recebem a popularidade do item no treino."""
        pop = torch.tensor(self.popularity, dtype=torch.float32)
        return lambda users: pop.expand(len(users), -1).clone()


def save_report(path: str, info: dict, reports: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Uma linha por split, com as colunas de identificação (modelo, seed, ...) na frente."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([{**info, "split": split, **metrics} for split, metrics in reports.items()])
    df.to_csv(path, index=False)
    return df


def summarize(results_dir: str, split: str = "test") -> pd.DataFrame:
    """Média, desvio e número de seeds por modelo, a partir dos CSVs de save_report."""
    files = [os.path.join(results_dir, f) for f in sorted(os.listdir(results_dir)) if f.endswith(".csv")
             and f != "summary.csv"]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["split"] == split]
    metrics = [c for c in df.columns if c not in ("model", "dataset", "seed", "split", "epoch")]
    summary = df.groupby("model")[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary.insert(0, "seeds", df.groupby("model")["seed"].nunique())
    summary.to_csv(os.path.join(results_dir, "summary.csv"))
    return summary
