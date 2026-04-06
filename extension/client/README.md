# 🔍 Vietnamese Fake News Detector Extension

Chrome/Opera extension để phát hiện tin giả trên Facebook.

## 📁 Cấu trúc

```
extension/
├── manifest.json      # Cấu hình extension (Manifest V3)
├── background.js      # Service worker - gọi API
├── content.js         # Inject vào Facebook - đọc DOM
├── styles.css         # CSS cho indicator badges
├── popup/
│   ├── popup.html     # UI popup khi click icon
│   ├── popup.js       # Logic popup
│   └── popup.css      # Style popup
└── icons/             # Icon extension (cần thêm)
    ├── icon16.png
    ├── icon48.png
    └── icon128.png
```

## 🚀 Cài đặt

### 1. Tạo icons (bắt buộc)
Cần thêm 3 file icon vào folder `icons/`:
- `icon16.png` (16x16 px)
- `icon48.png` (48x48 px)
- `icon128.png` (128x128 px)

### 2. Load vào Chrome/Opera

1. Mở `chrome://extensions/` (hoặc `opera://extensions/`)
2. Bật **Developer mode** (góc phải trên)
3. Click **Load unpacked**
4. Chọn folder `extension/`

### 3. Chạy server

```bash
cd ../server
python main.py
```

Server sẽ chạy tại `http://localhost:8000`

## 🔄 Luồng dữ liệu

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Facebook   │ →  │ content.js  │ →  │background.js│ →  │   Server    │
│    DOM      │    │ (đọc post)  │    │ (gọi API)   │    │  :8000      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      ↑                   ↑                  │                  │
      │                   │                  ↓                  ↓
      └───────────────────┴──────── response ←────────── predict
                    (hiển thị badge)
```

## 📌 Hai chế độ

| Chế độ | Mô tả | real_ratio |
|--------|-------|------------|
| **Feed** | Bài random trên newsfeed | Không tính (= 0.5) |
| **Group** | Bài trong nhóm | Tính từ lịch sử user |

## ⚠️ Lưu ý

- Extension chỉ hoạt động trên `facebook.com`
- Server phải đang chạy để extension hoạt động
- Facebook có thể thay đổi DOM → cần cập nhật selectors trong `content.js`
