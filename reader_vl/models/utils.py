from pathlib import Path

from huggingface_hub import hf_hub_download


def get_models_path() -> str:
    cache_dir = Path.home() / ".cache" / "reader-vl"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = hf_hub_download(
        repo_id="Aquos06/reader-vl", filename="best.pt", cache_dir=cache_dir
    )
    return model_path
