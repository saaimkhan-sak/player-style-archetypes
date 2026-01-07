from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Optional
import requests

DEFAULT_BASE = "https://api-web.nhle.com/v1"

class NHLClient:
    def __init__(self, base_url: str = DEFAULT_BASE, cache_dir: str = "data/raw", sleep_s: float = 0.15):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_s = sleep_s

    def _cache_path(self, rel: str) -> Path:
        # rel like: "season/20232024/schedule.json"
        return self.cache_dir / rel

    def get_json(self, endpoint: str, cache_relpath: str, force: bool = False) -> Any:
        """
        endpoint: "/schedule/2023-10-10" or "/club-stats/..." etc (no base url)
        cache_relpath: where to store under data/raw/
        """
        cache_path = self._cache_path(cache_relpath)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists() and not force:
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        url = f"{self.base_url}{endpoint}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        time.sleep(self.sleep_s)
        return data
