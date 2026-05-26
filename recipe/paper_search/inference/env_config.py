import os
from pathlib import Path


def load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


load_project_env()
