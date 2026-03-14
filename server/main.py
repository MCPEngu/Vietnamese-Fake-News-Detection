"""
Vietnamese Fake News Detection Server.
Pipeline: extension packet -> data processing -> embeddings -> trained model.
"""

from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data_processing.feature_extraction_functions import extract_required_features
from data_processing.text_cleaning_functions import prepare_text_for_embeddings
from model_runtime.embedding_loader import EmbeddingService
from model_runtime.model_loader import InferenceModel
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
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_ROOT = Path(__file__).resolve().parent

# Initialize components
user_history = UserHistoryManager(data_dir=str(SERVER_ROOT / "group_files"))
embedding_service = EmbeddingService(project_root=PROJECT_ROOT, enable_bert=True)
inference_model = InferenceModel(project_root=PROJECT_ROOT)

# ============================================================================
# MODELS
# ============================================================================
class PredictRequest(BaseModel):
    content_text: str
    timestamp: Optional[str] = None
    mode: str = "feed"  # "feed" hoặc "group"
    user_id: Optional[str] = None
    group_id: Optional[str] = None
    num_like: int = 0
    num_cmt: int = 0
    num_share: int = 0

class PredictResponse(BaseModel):
    label: int  # 0 = real, 1 = fake
    confidence: float
    features: Optional[dict] = None
    model_info: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    version: str

class GroupRequest(BaseModel):
    group_id: str


class GroupStatsResponse(BaseModel):
    group_id: str
    total_users: int
    total_posts: int
    total_fake: int

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
        if not request.content_text or len(request.content_text.strip()) < 5:
            raise HTTPException(status_code=400, detail="Content text too short")

        # 1) Prepare text for embedding
        text_bundle = prepare_text_for_embeddings(request.content_text)

        # 2) Group history flow
        fake_ratio = 0.0
        if request.mode == "group" and request.user_id and request.group_id:
            user_history.ensure_group_file(request.group_id)
            user_history.ensure_user(request.group_id, request.user_id)
            fake_ratio = user_history.get_fake_ratio(request.group_id, request.user_id)

        # 3) Extract 11 model features
        features = extract_required_features(
            content_text=text_bundle["text_raw_style"],
            timestamp=request.timestamp,
            num_like=request.num_like,
            num_cmt=request.num_cmt,
            num_share=request.num_share,
            fake_ratio=fake_ratio,
        )

        # 4) Embeddings (optional artifacts; call kept for full pipeline)
        embedding_bundle = embedding_service.encode(
            text_bert=text_bundle["text_bert"],
            text_tfidf=text_bundle["text_tfidf"],
        )

        # 5) Predict with trained model
        label, confidence, model_debug = inference_model.predict(features, embedding_bundle=embedding_bundle)

        # 6) Update history after prediction
        if request.mode == "group" and request.user_id and request.group_id:
            user_history.add_prediction(group_id=request.group_id, user_id=request.user_id, label=label)

        model_info: Dict[str, object] = {
            "model_config": "B1_hour_sin_cos_lightgbm",
            **model_debug,
            "bert_embedding_loaded": embedding_bundle["phobert_pretrain_embedding"] is not None,
            "tfidf_embedding_loaded": embedding_bundle["tfidf_embedding"] is not None,
        }

        return PredictResponse(
            label=label,
            confidence=confidence,
            features=features,
            model_info=model_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# GROUP ENDPOINTS (bảng background.js gọi khi vào/rời group)
# ============================================================================
@app.post("/group/enter")
async def group_enter(request: GroupRequest):
    """
    Thông báo extension bắt đầu theo dõi một group.
    Chưa có lógic đặc biệt - endpoint tồn tại để tránh 404.
    """
    user_history.ensure_group_file(request.group_id)
    return {"status": "ok", "group_id": request.group_id, "file_initialized": True}

@app.post("/group/leave")
async def group_leave(request: GroupRequest):
    """
    Thông báo extension rời khỏi một group.
    Chưa có lógic đặc biệt - endpoint tồn tại để tránh 404.
    """
    return {"status": "ok", "group_id": request.group_id}


@app.get("/group/{group_id}/stats", response_model=GroupStatsResponse)
async def group_stats(group_id: str):
    stats = user_history.get_group_stats(group_id)
    return GroupStatsResponse(**stats)

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
