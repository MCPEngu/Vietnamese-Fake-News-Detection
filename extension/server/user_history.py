"""
Group/User history manager.
One file per group, each file stores per-user counters:
user_id,num_post,num_fake
"""

import csv
import threading
from pathlib import Path
from typing import Dict


class UserHistoryManager:
    def __init__(self, data_dir: str = "group_files"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._lock = threading.Lock()

    def _sanitize_group_id(self, group_id: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in (group_id or "unknown"))

    def _get_group_file(self, group_id: str) -> Path:
        safe_group = self._sanitize_group_id(group_id)
        return self.data_dir / f"group_{safe_group}.csv"

    def ensure_group_file(self, group_id: str) -> Path:
        """Create group file if it does not exist."""
        path = self._get_group_file(group_id)
        if not path.exists():
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["user_id", "num_post", "num_fake"])
                writer.writeheader()
        return path

    def _load_group_data(self, group_id: str) -> Dict[str, Dict[str, int]]:
        if group_id in self._cache:
            return self._cache[group_id]

        path = self.ensure_group_file(group_id)
        group_data: Dict[str, Dict[str, int]] = {}

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                user_id = (row.get("user_id") or "").strip()
                if not user_id:
                    continue
                num_post = int(row.get("num_post") or 0)
                num_fake = int(row.get("num_fake") or 0)
                group_data[user_id] = {
                    "num_post": max(0, num_post),
                    "num_fake": max(0, num_fake),
                }

        self._cache[group_id] = group_data
        return group_data

    def _flush_group_data(self, group_id: str) -> None:
        path = self.ensure_group_file(group_id)
        group_data = self._cache.get(group_id, {})
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["user_id", "num_post", "num_fake"])
            writer.writeheader()
            for user_id, stats in sorted(group_data.items()):
                writer.writerow(
                    {
                        "user_id": user_id,
                        "num_post": int(stats.get("num_post", 0)),
                        "num_fake": int(stats.get("num_fake", 0)),
                    }
                )

    def ensure_user(self, group_id: str, user_id: str) -> None:
        with self._lock:
            group_data = self._load_group_data(group_id)
            if user_id not in group_data:
                group_data[user_id] = {"num_post": 0, "num_fake": 0}
                self._flush_group_data(group_id)

    def get_fake_ratio(self, group_id: str, user_id: str) -> float:
        with self._lock:
            group_data = self._load_group_data(group_id)
            user_stats = group_data.get(user_id)
            if not user_stats:
                return 0.0
            num_post = int(user_stats.get("num_post", 0))
            num_fake = int(user_stats.get("num_fake", 0))
            if num_post <= 0:
                return 0.0
            return round(num_fake / num_post, 6)

    def add_prediction(self, group_id: str, user_id: str, label: int) -> None:
        """Update counters after prediction. label: 0 real, 1 fake."""
        with self._lock:
            group_data = self._load_group_data(group_id)
            if user_id not in group_data:
                group_data[user_id] = {"num_post": 0, "num_fake": 0}

            group_data[user_id]["num_post"] += 1
            if int(label) == 1:
                group_data[user_id]["num_fake"] += 1

            self._flush_group_data(group_id)

    def get_group_stats(self, group_id: str) -> dict:
        with self._lock:
            group_data = self._load_group_data(group_id)
            total_users = len(group_data)
            total_posts = sum(u["num_post"] for u in group_data.values())
            total_fake = sum(u["num_fake"] for u in group_data.values())
            return {
                "group_id": group_id,
                "total_users": total_users,
                "total_posts": total_posts,
                "total_fake": total_fake,
            }
