# Vietnamese Fake News Detection Server

FastAPI backend cho extension, dùng pipeline inference thật:
`packet -> text processing -> embeddings -> LightGBM model -> update group history`.

## Cấu trúc mới

```
server/
├── main.py
├── user_history.py
├── requirements.txt
├── data_processing/
│   ├── feature_extraction_functions.py
│   └── text_cleaning_functions.py
├── model_runtime/
│   ├── embedding_loader.py
│   └── model_loader.py
└── group_files/
  └── group_<group_id>.csv
```

## 🚀 Cài đặt & Chạy

```bash
# Cài dependencies
pip install -r requirements.txt

# Chạy server
cd server
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
Phân tích bài đăng bằng model `B1_hour_sin_cos`.

**Request:**
```json
{
  "content_text": "Nội dung bài đăng...",
  "timestamp": "2024-01-15T19:30:00Z",
  "mode": "feed",
  "user_id": null,
  "group_id": null,
  "num_like": 100,
  "num_cmt": 20,
  "num_share": 10
}
```

**Response:**
```json
{
  "label": 0,
  "confidence": 0.85,
  "features": {
    "feat_avg_word_length": 4.5,
    "feat_comment_ratio": 0.2,
    "feat_digit_ratio": 0.05,
    "feat_fake_ratio": 0.1,
    "feat_hour_cos": 0.5,
    "feat_hour_sin": 0.866,
    "feat_like_ratio": 0.7,
    "feat_num_exclamation": 2,
    "feat_num_question": 1,
    "feat_num_sentences": 3,
    "feat_num_urls": 1
  },
  "model_info": {
    "model_config": "B1_hour_sin_cos_lightgbm",
    "input_vector_dim": 219,
    "model_num_features": 219
  }
}
```

### `POST /group/enter`
Tạo file group nếu chưa tồn tại.

### `POST /group/leave`
Đánh dấu rời group.

### `GET /group/{group_id}/stats`
Lấy thống kê lịch sử của group.

## Group history format

Mỗi group có 1 file CSV riêng tại `server/group_files/group_{id}.csv`:

```csv
user_id,num_post,num_fake
user_123,10,2
user_456,4,1
```
