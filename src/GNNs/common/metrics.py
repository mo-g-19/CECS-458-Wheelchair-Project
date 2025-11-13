#Quick sanity metrics for evaluation and logging
import numpy as np

def auc_at_k(labels, scores, k=10):
    #tiny, approximate AIC@k for sanity checks
    idx = np.argsort(-scores)[:k]
    return float(np.mean(labels[idx]))