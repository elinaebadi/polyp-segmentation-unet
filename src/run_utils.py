
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from config import OUTPUT_DIR


def create_run_paths(
    experiment_name: str,
    base_dir: Path | None = None
):
    """
    Create a unique directory for a training run.

    Returns:
        checkpoint_path, history_path
    """

    if not experiment_name.replace("_", "").isalnum():
        raise ValueError("Invalid experiment name.")

    if base_dir is None:
        base_dir = OUTPUT_DIR / "runs"

    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid4().hex[:8]
    )

    run_dir = Path(base_dir) / experiment_name / run_id

    # Never reuse an existing experiment directory
    run_dir.mkdir(parents=True, exist_ok=False)

    checkpoint_path = run_dir / "best.pth"
    history_path = run_dir / "history.csv"

    return checkpoint_path, history_path
