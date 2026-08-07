import sys

from .main import run_pipeline


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(dataset_path=dataset_path)
