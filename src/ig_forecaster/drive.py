from __future__ import annotations

import os
import sys
from pathlib import Path


def mount_google_drive() -> None:
    drive_path = Path("/content/drive")
    if drive_path.exists():
        return

    if "google.colab" not in sys.modules:
        return

    try:
        from google.colab import drive
    except Exception:
        return

    os.makedirs("/content/drive", exist_ok=True)
    drive.mount("/content/drive", force_remount=True)
