import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from midterm.main import run_train


if __name__ == "__main__":
    args = SimpleNamespace(
        epochs=None,
        batch_size=None,
        lr=None,
        no_augment=False,
        resplit=False,
        output=None,
        param_log_interval=0,
    )
    run_train(args)
