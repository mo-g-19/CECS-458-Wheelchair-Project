#Quick sanity metrics for evaluation and logging
# src/GNNs/common/metrics.py
import numpy as np
import torch

# ------------------------
# Coverage / degrees (unchanged helpers)
# ------------------------
def _deg_from_edge_index(edge_index, num_nodes):
    deg = torch.zeros(num_nodes, dtype=torch.long)
    deg.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
    deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), dtype=torch.long))
    return deg

def _feat_coverage(x: torch.Tensor):
    if x is None or x.numel() == 0:
        return dict(dim=0, nnz_ratio=0.0, zero_cols=0, zero_rows=0)
    nnz_ratio = float((x != 0).float().mean().item())
    col_is_zero = (x == 0).all(dim=0)
    row_is_zero = (x == 0).all(dim=1)
    return dict(
        dim=x.size(1),
        nnz_ratio=nnz_ratio,
        zero_cols=int(col_is_zero.sum().item()),
        zero_rows=int(row_is_zero.sum().item()),
    )

def compute_stats(hetero_data):
    s = dict(nodes={}, degrees={}, edges={}, features={})
    for ntype in hetero_data.node_types:
        num = hetero_data[ntype].num_nodes
        s["nodes"][ntype] = int(num)
        x = getattr(hetero_data[ntype], "x", None)
        s["features"][ntype] = _feat_coverage(x)

    for rel in hetero_data.edge_types:
        src, relname, dst = rel
        ei = hetero_data[rel].edge_index
        s["edges"][relname if src == dst else f"{src}-{relname}-{dst}"] = int(ei.size(1))

        deg_src = _deg_from_edge_index(ei, hetero_data[src].num_nodes)
        key_src = f"{src}::via::{relname}"
        s["degrees"][key_src] = dict(
            mean=float(deg_src.float().mean().item()),
            p95=float(torch.quantile(deg_src.float(), 0.95).item()),
            max=int(deg_src.max().item()),
            iso_ratio=float((deg_src == 0).float().mean().item()),
        )
        if src != dst:
            deg_dst = _deg_from_edge_index(ei.flip(0), hetero_data[dst].num_nodes)
            key_dst = f"{dst}::via::{relname}"
            s["degrees"][key_dst] = dict(
                mean=float(deg_dst.float().mean().item()),
                p95=float(torch.quantile(deg_dst.float(), 0.95).item()),
                max=int(deg_dst.max().item()),
                iso_ratio=float((deg_dst == 0).float().mean().item()),
            )
    return s

def coverage(stats):
    out = {"nodes": {}, "features": {}, "degrees": {}, "edges": stats["edges"]}
    for ntype, n in stats["nodes"].items():
        out["nodes"][ntype] = dict(count=n)
    for ntype, f in stats["features"].items():
        out["features"][ntype] = dict(
            dim=f["dim"],
            nnz_pct=round(100 * f["nnz_ratio"], 2),
            zero_cols=f["zero_cols"],
            zero_rows=f["zero_rows"],
        )
    for k, d in stats["degrees"].items():
        out["degrees"][k] = dict(
            mean=round(d["mean"], 3),
            p95=round(d["p95"], 3),
            max=d["max"],
            isolated_pct=round(100 * d["iso_ratio"], 2),
        )
    return out

# ------------------------
# Ranking / classification metrics
# ------------------------
def _sorted_labels(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    order = np.argsort(-scores, kind="mergesort")  # stable for ties
    return labels[order], scores[order]

def roc_auc(labels, scores):
    """Full ROC AUC (no sklearn)."""
    y, s = _sorted_labels(labels, scores)
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return 1.0  # degenerate but defined

    # Walk down the ranked list and trace ROC
    tp = 0; fp = 0
    prev_s = None
    fprs = [0.0]; tprs = [0.0]
    for yi, si in zip(y, s):
        if prev_s is not None and si != prev_s:
            fprs.append(fp / N)
            tprs.append(tp / P)
        if yi == 1: tp += 1
        else:       fp += 1
        prev_s = si
    fprs.append(fp / N); tprs.append(tp / P)

    # Trapezoid
    area = 0.0
    for i in range(1, len(fprs)):
        area += (fprs[i] - fprs[i-1]) * (tprs[i] + tprs[i-1]) / 2.0
    return float(area)

def auc_at_k(labels, scores, k=10, normalize=True):
    """
    Normalized partial ROC-AUC computed using only the top-k ranked prefix.
    - If normalize=True, divide by the maximum possible area in [0, FPR_k] so result ∈ [0,1].
    - If you want a simpler 'precision in top-k', use precision_at_k instead.
    """
    y, s = _sorted_labels(labels, scores)
    k = int(min(k, len(y)))
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0 or k == 0:
        return 1.0

    tp = 0; fp = 0
    fprs = [0.0]; tprs = [0.0]
    prev_s = None
    seen = 0
    for yi, si in zip(y, s):
        if seen >= k: break
        if prev_s is not None and si != prev_s:
            fprs.append(fp / N)
            tprs.append(tp / P)
        if yi == 1: tp += 1
        else:       fp += 1
        prev_s = si
        seen += 1
    fprs.append(fp / N); tprs.append(tp / P)

    # Trapezoid over [0, FPR_k]
    area = 0.0
    for i in range(1, len(fprs)):
        area += (fprs[i] - fprs[i-1]) * (tprs[i] + tprs[i-1]) / 2.0

    if not normalize:
        return float(area)
    fpr_k = fprs[-1]
    return float(area / max(fpr_k, 1e-12))  # normalized partial AUC ∈ [0,1]

def precision_at_k(labels, scores, k=10):
    y, _ = _sorted_labels(labels, scores)
    k = int(min(k, len(y)))
    return float(y[:k].mean()) if k > 0 else 0.0

def recall_at_k(labels, scores, k=10):
    y, _ = _sorted_labels(labels, scores)
    P = y.sum()
    if P == 0:
        return 1.0
    k = int(min(k, len(y)))
    return float(y[:k].sum() / P)

def ndcg_at_k(labels, scores, k=10):
    y, _ = _sorted_labels(labels, scores)
    k = int(min(k, len(y)))
    if k == 0:
        return 0.0
    gains = (2 ** y[:k] - 1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))

    # Ideal DCG
    y_sorted = np.sort(y)[::-1]
    gains_i = (2 ** y_sorted[:k] - 1)
    idcg = float(np.sum(gains_i * discounts))
    return dcg / idcg if idcg > 0 else 1.0

def hit_rate_at_k(labels, scores, k=10):
    y, _ = _sorted_labels(labels, scores)
    k = int(min(k, len(y)))
    return float(y[:k].sum() > 0)
