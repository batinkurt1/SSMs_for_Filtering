import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
INNER_SRC = PROJECT_ROOT / "src"
if str(INNER_SRC) not in sys.path:
    sys.path.insert(0, str(INNER_SRC))

from src.data_collection.generate_training_data import generate_training_data
from src.training.train import train
from src.testing.test import test
from src.plotting.plot_results import plot
from src.helpers.cfg_loader import load_yaml


CFG = load_yaml("configs/main.yaml")
generate_data_flag = bool(CFG["generate_data_flag"])
train_flag = bool(CFG["train_flag"])
test_flag = bool(CFG["test_flag"])
plot_flag = bool(CFG["plot_flag"])

if __name__ == "__main__":
    if generate_data_flag:
        generate_training_data()
    if train_flag:
        train()
    if test_flag:
        test()
    if plot_flag:
        plot()