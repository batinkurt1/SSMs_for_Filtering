import pickle
from pathlib import Path

def load_pickle(path):
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)