# 🖥️ Vietnamese Fake News Detection Server

FastAPI backend xử lý prediction từ extension.

## 📁 Cấu trúc

```
server/
├── main.py              # FastAPI app + endpoints
├── feature_extractor.py # Trích xuất features từ text
├── user_history.py      # Quản lý lịch sử user (CSV)
├── requirements.txt     # Dependencies
└── data/                # Thư mục lưu data (auto-created)
    └── user_history/    # CSV files cho mỗi nhóm
```

## 🚀 Cài đặt & Chạy

```bash
# Cài dependencies
pip install -r requirements.txt

# Chạy server
python main.py
```

Server sẽ chạy tại:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

## 📡 API Endpoints

### `GET /health`
Check server status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### `POST /predict`
Phân tích bài đăng.

**Request:**
```json
{
  "content_text": "Nội dung bài đăng...",
  "timestamp": "2024-01-15T19:30:00Z",
  "mode": "feed",
  "user_id": null,
  "group_id": null
}
```

**Response:**
```json
{
  "label": 0,
  "confidence": 0.85,
  "features": {
    "feat_num_sentences": 3,
    "feat_num_exclamation": 2,
    "feat_num_question": 1,
    "feat_avg_word_length": 4.5,
    "feat_num_urls": 1,
    "feat_digit_ratio": 0.05,
    "feat_is_evening": 1,
    "feat_real_ratio": 0.5
  }
}
```

## 📊 Features được trích xuất

| Feature | Nguồn | Mô tả |
|---------|-------|-------|
| `feat_num_sentences` | content_text | Số câu |
| `feat_num_exclamation` | content_text | Số dấu ! |
| `feat_num_question` | content_text | Số dấu ? |
| `feat_avg_word_length` | content_text | Độ dài TB từ |
| `feat_num_urls` | content_text | Số URL |
| `feat_digit_ratio` | content_text | Tỷ lệ số |
| `feat_is_evening` | timestamp | 18h-22h? |
| `feat_real_ratio` | user_history | Tỷ lệ tin thật của user |

## 📂 User History (Chế độ Group)

Mỗi nhóm có 1 file CSV riêng tại `data/user_history/group_{id}.csv`:

```csv
user_id,label,timestamp
user_123,0,2024-01-15T10:30:00
user_123,1,2024-01-15T11:45:00
user_456,0,2024-01-15T12:00:00
```

## 🔧 TODO

- [ ] Tích hợp TF-IDF model thực
- [ ] Tích hợp BERT embeddings
- [ ] Load classifier đã train
- [ ] Thêm caching cho performance
- [ ] Thêm rate limiting
- [ ] Thêm authentication
