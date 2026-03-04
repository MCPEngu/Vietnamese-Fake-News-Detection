"""
User History Manager
Quản lý lịch sử user cho từng nhóm
"""

import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading


class UserHistoryManager:
    """
    Quản lý lịch sử bài đăng của user theo nhóm
    
    - Mỗi nhóm có 1 file CSV riêng
    - Lưu: user_id, label, timestamp
    - Tính real_ratio = num_real / total_posts
    """
    
    def __init__(self, data_dir: str = "data/user_history"):
        """
        Args:
            data_dir: Thư mục lưu các file CSV
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache để tránh đọc file liên tục
        self._cache = {}
        self._lock = threading.Lock()
    
    def _get_csv_path(self, group_id: str) -> Path:
        """Lấy đường dẫn file CSV cho group"""
        # Sanitize group_id để làm tên file
        safe_name = "".join(c if c.isalnum() else "_" for c in group_id)
        return self.data_dir / f"group_{safe_name}.csv"
    
    def _load_group_data(self, group_id: str) -> dict:
        """
        Load dữ liệu của group từ CSV
        
        Returns:
            dict mapping user_id -> {"real": int, "fake": int, "total": int}
        """
        csv_path = self._get_csv_path(group_id)
        
        if group_id in self._cache:
            return self._cache[group_id]
        
        data = {}
        
        if csv_path.exists():
            try:
                with open(csv_path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        user_id = row['user_id']
                        label = int(row['label'])
                        
                        if user_id not in data:
                            data[user_id] = {"real": 0, "fake": 0, "total": 0}
                        
                        data[user_id]["total"] += 1
                        if label == 0:
                            data[user_id]["real"] += 1
                        else:
                            data[user_id]["fake"] += 1
            except Exception as e:
                print(f"[UserHistory] Error loading {csv_path}: {e}")
        
        self._cache[group_id] = data
        return data
    
    def get_real_ratio(self, group_id: str, user_id: str) -> float:
        """
        Tính tỷ lệ tin thật của user trong group
        
        Args:
            group_id: ID của nhóm
            user_id: ID của user
            
        Returns:
            float trong khoảng [0, 1], mặc định 0.5 nếu chưa có data
        """
        with self._lock:
            data = self._load_group_data(group_id)
            
            if user_id not in data or data[user_id]["total"] == 0:
                return 0.5  # Default value khi chưa có lịch sử
            
            user_data = data[user_id]
            return round(user_data["real"] / user_data["total"], 4)
    
    def add_record(self, group_id: str, user_id: str, label: int) -> None:
        """
        Thêm record mới vào history
        
        Args:
            group_id: ID của nhóm
            user_id: ID của user
            label: 0 (real) hoặc 1 (fake)
        """
        with self._lock:
            csv_path = self._get_csv_path(group_id)
            
            # Cập nhật cache
            if group_id not in self._cache:
                self._load_group_data(group_id)
            
            if user_id not in self._cache[group_id]:
                self._cache[group_id][user_id] = {"real": 0, "fake": 0, "total": 0}
            
            self._cache[group_id][user_id]["total"] += 1
            if label == 0:
                self._cache[group_id][user_id]["real"] += 1
            else:
                self._cache[group_id][user_id]["fake"] += 1
            
            # Append to CSV
            file_exists = csv_path.exists()
            
            try:
                with open(csv_path, 'a', encoding='utf-8', newline='') as f:
                    fieldnames = ['user_id', 'label', 'timestamp']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                    
                    writer.writerow({
                        'user_id': user_id,
                        'label': label,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                print(f"[UserHistory] Error writing to {csv_path}: {e}")
    
    def get_group_stats(self, group_id: str) -> dict:
        """
        Lấy thống kê của group
        
        Returns:
            {"total_users": int, "total_posts": int, "real_posts": int, "fake_posts": int}
        """
        with self._lock:
            data = self._load_group_data(group_id)
            
            total_users = len(data)
            total_posts = sum(u["total"] for u in data.values())
            real_posts = sum(u["real"] for u in data.values())
            fake_posts = sum(u["fake"] for u in data.values())
            
            return {
                "total_users": total_users,
                "total_posts": total_posts,
                "real_posts": real_posts,
                "fake_posts": fake_posts
            }
    
    def clear_cache(self, group_id: Optional[str] = None) -> None:
        """Clear cache (sử dụng khi cần reload data)"""
        with self._lock:
            if group_id:
                self._cache.pop(group_id, None)
            else:
                self._cache.clear()


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    manager = UserHistoryManager(data_dir="data/test_history")
    
    # Thêm một số records test
    test_group = "test_group_123"
    
    manager.add_record(test_group, "user_1", 0)  # real
    manager.add_record(test_group, "user_1", 0)  # real
    manager.add_record(test_group, "user_1", 1)  # fake
    manager.add_record(test_group, "user_2", 1)  # fake
    manager.add_record(test_group, "user_2", 1)  # fake
    
    print("=" * 50)
    print("User real ratios:")
    print("=" * 50)
    print(f"  user_1: {manager.get_real_ratio(test_group, 'user_1')}")  # 0.6667
    print(f"  user_2: {manager.get_real_ratio(test_group, 'user_2')}")  # 0.0
    print(f"  user_3 (new): {manager.get_real_ratio(test_group, 'user_3')}")  # 0.5 (default)
    
    print("\nGroup stats:")
    print("=" * 50)
    stats = manager.get_group_stats(test_group)
    for key, value in stats.items():
        print(f"  {key}: {value}")
