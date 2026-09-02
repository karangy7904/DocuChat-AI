from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    model_id: str = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    backend: str = os.getenv("GENERATOR_BACKEND", "transformers")
    top_k: int = int(os.getenv("TOP_K", "3"))
    adapter_path: Path = ROOT / "models" / "docuchat-lora"


settings = Settings()

