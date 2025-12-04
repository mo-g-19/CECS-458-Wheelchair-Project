import os, random, numpy as np, torch
def _seed():
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
