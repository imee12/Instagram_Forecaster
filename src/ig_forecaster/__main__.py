import sys

from .main import run_workflow


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_workflow(dataset_path=dataset_path)
    if result.get("errors"):
        raise SystemExit("; ".join(result["errors"]))
