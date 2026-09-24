"""
tasks/hf_download.py
Load a HuggingFace dataset split as a list of dict rows.

Uses `datasets.load_dataset` when the package is installed; otherwise downloads the parquet
files the Hub publishes for every dataset (needs only requests + pandas + pyarrow).
"""

import io
from typing import List

import requests

_API = "https://huggingface.co/api/datasets/{repo}/parquet/{config}/{split}"


def _to_python(x):
    """numpy arrays / pandas scalars inside parquet rows -> plain lists and values."""
    if hasattr(x, "tolist"):
        return _to_python(x.tolist())
    if isinstance(x, dict):
        return {k: _to_python(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_python(v) for v in x]
    return x


def load_rows(repo: str, config: str = "default", split: str = "test") -> List[dict]:
    try:
        from datasets import load_dataset
        name = None if config == "default" else config
        return [dict(r) for r in load_dataset(repo, name, split=split)]
    except ImportError:
        pass
    import pandas as pd
    urls = requests.get(_API.format(repo=repo, config=config, split=split), timeout=60).json()
    if not isinstance(urls, list) or not urls:
        raise RuntimeError(f"no parquet files for {repo}/{config}/{split}: {urls}")
    rows = []
    for url in urls:
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        df = pd.read_parquet(io.BytesIO(resp.content))
        rows.extend(_to_python(r) for r in df.to_dict(orient="records"))
    return rows
