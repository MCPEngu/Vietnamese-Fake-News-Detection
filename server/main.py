"""
Vietnamese Fake News Detection Server
FastAPI backend xử lý prediction từ extension
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from datetime import datetime
import random

from feature_extractor import FeatureExtractor
from user_history import UserHistoryManager

# ============================================================================
# APP SETUP
# ============================================================================
app = FastAPI(
    title="Vietnamese Fake News Detection API",
    description="API phát hiện tin giả tiếng Việt",
    version="1.0.0"
)

# CORS - cho phép extension gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
feature_extractor = FeatureExtractor()
user_history = UserHistoryManager()

# ============================================================================
# MODELS
# ============================================================================
class PredictRequest(BaseModel):
    content_text: str
    timestamp: Optional[str] = None
    mode: str = "feed"  # "feed" hoặc "group"
    user_id: Optional[str] = None
    group_id: Optional[str] = None

class PredictResponse(BaseModel):
    label: int  # 0 = real, 1 = fake
    confidence: float
    features: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    version: str

# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server status"""
    return HealthResponse(
        status="healthy",
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Phân tích bài đăng và trả về prediction
    
    - **content_text**: Nội dung bài đăng
    - **timestamp**: Thời gian đăng (ISO format)
    - **mode**: "feed" (mặc định) hoặc "group"
    - **user_id**: ID người đăng (chỉ cần khi mode=group)
    - **group_id**: ID nhóm (chỉ cần khi mode=group)
    """
    try:
        # Validate input
        if not request.content_text or len(request.content_text) < 10:
            raise HTTPException(status_code=400, detail="Content text too short")
        
        # Extract features từ text
        features = feature_extractor.extract(
            content_text=request.content_text,
            timestamp=request.timestamp
        )
        
        # Nếu ở chế độ group, tính real_ratio từ history
        if request.mode == "group" and request.user_id and request.group_id:
            real_ratio = user_history.get_real_ratio(
                group_id=request.group_id,
                user_id=request.user_id
            )
            features["feat_real_ratio"] = real_ratio
        else:
            features["feat_real_ratio"] = 0.5  # Default value cho feed mode
        
        # TODO: Thay bằng model thực khi ready
        # Hiện tại trả kết quả giả để test luồng
        label, confidence = mock_predict(features)
        
        # Nếu ở chế độ group, lưu kết quả vào history
        if request.mode == "group" and request.user_id and request.group_id:
            user_history.add_record(
                group_id=request.group_id,
                user_id=request.user_id,
                label=label
            )
        
        return PredictResponse(
            label=label,
            confidence=confidence,
            features=features
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MOCK PREDICTION (TODO: Thay bằng model thực)
# ============================================================================
def mock_predict(features: dict) -> tuple[int, float]:
    """
    Mock prediction - trả kết quả ngẫu nhiên để test luồng
    
    TODO: Thay bằng model thực khi ready:
    1. Load TF-IDF model và transform text
    2. Load BERT model và get embeddings
    3. Combine features
    4. Load classifier và predict
    """
    # Dựa vào một số heuristics đơn giản để có kết quả realistic hơn
    fake_score = 0.0
    
    # Nhiều dấu chấm than → nghi ngờ hơn
    fake_score += min(features.get("feat_num_exclamation", 0) * 0.1, 0.3)
    
    # Nhiều URL → nghi ngờ hơn
    fake_score += min(features.get("feat_num_urls", 0) * 0.15, 0.3)
    
    # Post buổi tối → nghi ngờ hơn một chút
    if features.get("feat_is_evening", 0) == 1:
        fake_score += 0.1
    
    # Thêm random noise
    fake_score += random.uniform(-0.2, 0.2)
    
    # Clamp to [0, 1]
    fake_score = max(0.0, min(1.0, fake_score))
    
    # Quyết định label
    if fake_score > 0.5:
        return 1, fake_score
    else:
        return 0, 1 - fake_score

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Vietnamese Fake News Detection Server")
    print("=" * 60)
    print("📍 API Docs: http://localhost:8000/docs")
    print("📍 Health Check: http://localhost:8000/health")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
