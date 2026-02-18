# Vietnamese-Fake-News-Detection
Xây dựng model dự đoán thông tin giả mạng xã hội tiếng Việt

## 📋 Thông tin dự án
- **Python version**: 3.12.0
- **Virtual environment**: `.venv/`
- **Dataset**: 4736 bài viết (Real: 3929, Fake: 807)
- **SOTA to beat**: AUC 96.47% (Public), 95.21% (Private)

---

## 🎯 MỤC TIÊU NGHIÊN CỨU

Xây dựng hệ thống phát hiện tin giả tiếng Việt với:
1. Feature engineering toàn diện (44 features mới) + đánh giá thống kê
2. So sánh nhiều kiến trúc model (Single, Ensemble, Attention Fusion, Cross-Attention)
3. Error Analysis → Iterative Improvement
4. Vượt trội so với SOTA (AUC > 95.21%)

---

## 📁 CẤU TRÚC THƯ MỤC

```
Vietnamese-Fake-News-Detection/
├── data/
│   ├── raw/                        # Data gốc
│   │   └── data.csv                # 4736 rows, 26 columns
│   ├── processed/                  # Data đã xử lý
│   │   └── data_with_features.csv  # 4736 rows, 70 columns
│   ├── encoded/                    # Embeddings (TODO)
│   └── reduced/                    # Reduced embeddings (TODO)
│
├── src/
│   ├── preprocessing/              # Text cleaning & Feature engineering
│   │   ├── __init__.py
│   │   ├── text_cleaning.py        # clean_for_tfidf, clean_for_bert
│   │   ├── feature_engineering.py  # create_all_features
│   │   └── run_preprocessing.py    # Main pipeline
│   │
│   ├── analysis/                   # Feature analysis (TODO)
│   │   └── feature_analysis.ipynb
│   │
│   ├── embedding/                  # Text embeddings (TODO)
│   │   ├── tfidf_embedding.py
│   │   └── bert_embedding.py
│   │
│   ├── modeling/                   # Model training (TODO)
│   │   ├── single_models.py
│   │   ├── fusion_models.py
│   │   └── attention_fusion.py
│   │
│   └── evaluation/                 # Evaluation (TODO)
│       └── metrics.py
│
├── models/                         # Saved models (TODO)
├── results/                        # Experiment results (TODO)
├── requirements.txt
└── README.md
```

---

## 🔬 KẾ HOẠCH NGHIÊN CỨU CHI TIẾT

### PHASE 1: DATA & FEATURE ENGINEERING

#### 1.1 Feature Extraction
| Nhóm | Features | Nguồn |
|------|----------|-------|
| **Text-based** | num_char, num_word, num_emoji, num_url, num_hashtag, uppercase_ratio | post_message |
| **User-based** | num_fake, num_real, post_ratio, num_post | user_name + label |
| **Engagement** | num_like, num_cmt, num_share, like_ratio, cmt_ratio | interactions |
| **Temporal** | hour, weekday, month, is_weekend, is_night | timestamp |

#### 1.2 Feature Analysis (Statistical)
- [ ] **Cohen's d**: Đo effect size (|d| > 0.8 = large effect)
- [ ] **KS Statistic**: Đo khác biệt phân phối
- [ ] **T-test p-value**: Kiểm định ý nghĩa thống kê
- [ ] **Correlation Matrix**: Loại features trùng lặp (r > 0.9)
- [ ] **Visualization**: Boxplot, KDE plot, Heatmap

#### 1.3 Feature Selection
- [ ] Loại features có Cohen's d < 0.2 (negligible effect)
- [ ] Loại 1 trong 2 features có correlation > 0.9
- [ ] Random Forest Feature Importance

---

### PHASE 2: TEXT PROCESSING

#### 2.1 Text Preprocessing
- [ ] Remove noise: URL, HTML, emoji
- [ ] Normalize: lowercase, unicode
- [ ] Word segmentation: Giữ dấu `_` (data đã có sẵn)
- [ ] **Data Augmentation** (optional): Back-translation, synonym replacement

#### 2.2 Text Embedding - DANH SÁCH ĐẦY ĐỦ

##### 📊 BẢNG TỔNG HỢP CÁC EMBEDDING

| # | Model | Parameters | Dim | Max Tokens | Language | HuggingFace ID |
|---|-------|------------|-----|------------|----------|----------------|
| **CLASSICAL** |
| 1 | TF-IDF | 0 | ~300-5000 | ∞ | Any | scikit-learn |
| **VIETNAMESE-SPECIFIC** |
| 5 | PhoBERT-large | **370M** | 1024 | 256 | VI | `vinai/phobert-large` |
| 6 | PhoBERT-v2 | **135M** | 768 | 256 | VI | `vinai/phobert-base-v2` |
| 7 | ViSoBERT | **135M** | 768 | 256 | VI | `uitnlp/visobert` |



#### 3.4 Quy trình giảm chiều
```
TF-IDF (5000) ──► Truncated SVD ──► 300 chiều ─┐
                                                │
BERT (768) ────► PCA (optional) ──► 256 chiều ─┼──► Concat ──► Model
                                                │
Features (20) ─► Giữ nguyên ──────► 20 chiều ──┘
```

#### ⚠️ LƯU Ý QUAN TRỌNG
1. **Fit SVD/PCA CHỈ trên TRAIN set** → Transform cả train và test
2. **Tree-based models (XGBoost, LightGBM)**: Thường **KHÔNG CẦN** giảm chiều nhiều
3. **Neural networks**: Có thể cần giảm chiều để tránh overfitting

#### 3.5 Experiments giảm chiều
| Experiment | Mô tả |
|------------|-------|
| **Baseline** | Không giảm chiều |
| **SVD cho TF-IDF** | So sánh 100 vs 200 vs 300 chiều |
| **PCA cho BERT** | So sánh 128 vs 256 vs giữ nguyên 768 |

---

### PHASE 4: MODEL ARCHITECTURES

#### 4.1 Machine Learning Algorithms

##### 📊 DANH SÁCH ĐẦY ĐỦ CÁC THUẬT TOÁN

| # | Algorithm | Parameters | Ưu điểm | Nhược điểm | Dùng trong dự án |
|---|-----------|------------|---------|------------|------------------|
| **BASELINE** |
| 1 | **Logistic Regression** | ~1K | Nhanh, interpretable, baseline chuẩn | Yếu với non-linear | ✅ **Baseline** |
| **BOOSTING (Main Focus)** |
| 2 | **LightGBM** | ~100K | Nhanh nhất trong boosting, memory efficient | Cần tuning | ✅ **Main** |
| 3 | **XGBoost** | ~100K | SOTA tabular, robust, regularization tốt | Chậm hơn LightGBM | ✅ **Main** |
| 4 | **CatBoost** | ~100K | Tự xử lý categorical, ít overfitting | Chậm hơn LightGBM | ✅ **Main** |

##### 📊 SO SÁNH CHI TIẾT BOOSTING ALGORITHMS

| Thuật toán | Speed | Accuracy | Memory | Categorical | GPU Support |
|------------|-------|----------|--------|-------------|-------------|
| **LightGBM** | ⭐⭐⭐ Nhanh nhất | ⭐⭐⭐ | ⭐⭐⭐ Ít nhất | Cần encoding | ✅ Có |
| **XGBoost** | ⭐⭐ | ⭐⭐⭐ Robust nhất | ⭐⭐ | Cần encoding | ✅ Có |
| **CatBoost** | ⭐ Chậm nhất | ⭐⭐⭐ | ⭐ Nhiều nhất | ✅ Tự động | ✅ Có |

##### 📊 SO SÁNH NEURAL NETWORKS

| Model | Training Time | GPU cần | Khi nào dùng |
|-------|---------------|---------|--------------|
| **MLP** | 5-10 phút | Không cần (có thì nhanh hơn) | Fusion architectures, baseline NN |
| **Fine-tuned PhoBERT** | 30-60 phút | 4-8 GB VRAM | Text classification SOTA |
| **Fine-tuned XLM-RoBERTa** | 1-2 giờ | 8-12 GB VRAM | Multilingual, nếu PhoBERT không đủ |

#### 4.2 Model Architectures

##### 📊 SINGLE MODELS
Dùng 1 model duy nhất với tất cả features concat:
```
[Text Embedding] + [Handcrafted Features] → Concat → Single Model → Prediction
```

| Model | Input | Đặc điểm |
|-------|-------|----------|
| Logistic Regression | All features concat | Baseline |
| LightGBM | All features concat | Fast |
| XGBoost | All features concat | Robust |
| CatBoost | All features concat | Auto-categorical |
| MLP | All features concat | Neural baseline |

##### 📊 FUSION ARCHITECTURES

| # | Kiến trúc | Mô tả | Complexity | Dùng |
|---|-----------|-------|------------|------|
| 1 | **Early Fusion** | Concat tất cả features → 1 model | ⭐ | ✅ |
| 2 | **Late Fusion** | Train riêng từng branch → average/vote predictions | ⭐⭐ | ✅ |
| 3 | **Stacking** | Train riêng → dùng meta-learner kết hợp | ⭐⭐ | ✅ |
| 4 | **Attention Fusion** | Late Fusion + learned weights (PROPOSED) | ⭐⭐⭐ | ✅ |
| 5 | **Gated Fusion** | Gates điều khiển information flow | ⭐⭐⭐ | ✅ |
| 6 | **Multi-head Attention Fusion** | Nhiều attention heads | ⭐⭐⭐ | ✅ |
| 7 | **Cross-Attention Fusion** | Modalities attend to each other | ⭐⭐⭐⭐ | ✅ |
| 8 | **Hierarchical Stack + Attention** | Branch stacking → Attention fusion | ⭐⭐⭐⭐ | ✅ |

##### 📊 CHI TIẾT TỪNG FUSION ARCHITECTURE

---

**1. EARLY FUSION (Đơn giản nhất)**

```
Text (300)     ─┐
User (5)       ─┼─► Concat (325) ─► XGBoost ─► Prediction
Engagement (5) ─┘
Temporal (5)   ─┘
```

| Thành phần | Giải thích |
|------------|------------|
| **Concat** | Nối tất cả vectors thành 1 vector dài |
| **Single Model** | 1 model duy nhất học trên tất cả features |

**Ưu điểm**: Đơn giản, nhanh, dễ implement
**Nhược điểm**: Không phân biệt importance của từng modality
**Code concept**:
```python
X = np.concatenate([text_embed, user_feat, engagement_feat], axis=1)
model = XGBClassifier()
model.fit(X, y)     
```

---

**2. LATE FUSION (Average/Voting)**

```
Text (300)     ─► XGBoost₁ ─► P₁ ─┐
User (5)       ─► XGBoost₂ ─► P₂ ─┼─► Average ─► Final Prediction
Engagement (5) ─► XGBoost₃ ─► P₃ ─┘
```

| Thành phần | Giải thích |
|------------|------------|
| **Branch Models** | Mỗi modality có model riêng |
| **P₁, P₂, P₃** | Probability predictions từ mỗi branch |
| **Average** | Trung bình cộng: P_final = (P₁ + P₂ + P₃) / 3 |

**Ưu điểm**: Mỗi modality được xử lý riêng, robust
**Nhược điểm**: Coi tất cả branches quan trọng như nhau
**Code concept**:
```python
model_text = XGBClassifier().fit(text_embed, y)
model_user = XGBClassifier().fit(user_feat, y)
model_eng = XGBClassifier().fit(engagement_feat, y)

P1 = model_text.predict_proba(text_embed)[:, 1]
P2 = model_user.predict_proba(user_feat)[:, 1]
P3 = model_eng.predict_proba(engagement_feat)[:, 1]

P_final = (P1 + P2 + P3) / 3
```

---

**3. STACKING (Meta-learner)**

```
Text (300)     ─► XGBoost₁ ─► P₁ ─┐
User (5)       ─► XGBoost₂ ─► P₂ ─┼─► [P₁, P₂, P₃] ─► Meta-Learner ─► Final
Engagement (5) ─► XGBoost₃ ─► P₃ ─┘                    (LR hoặc XGB)
```

| Thành phần | Giải thích |
|------------|------------|
| **Base Models** | XGBoost₁, XGBoost₂, XGBoost₃ - train trên từng modality |
| **Out-of-fold predictions** | Dùng CV để tạo predictions không bị leak |
| **Meta-learner** | Model thứ 2 học cách combine predictions |

**Ưu điểm**: Meta-learner tự học weights tối ưu
**Nhược điểm**: Cần CV cẩn thận để tránh overfitting
**Code concept**:
```python
# Step 1: Get out-of-fold predictions
P1_oof = cross_val_predict(XGBClassifier(), text_embed, y, cv=5, method='predict_proba')[:, 1]
P2_oof = cross_val_predict(XGBClassifier(), user_feat, y, cv=5, method='predict_proba')[:, 1]
P3_oof = cross_val_predict(XGBClassifier(), engagement_feat, y, cv=5, method='predict_proba')[:, 1]

# Step 2: Train meta-learner
meta_features = np.column_stack([P1_oof, P2_oof, P3_oof])
meta_model = LogisticRegression().fit(meta_features, y)
```

---

**4. ATTENTION FUSION (PROPOSED)**

```
Text (300)     ─► Dense(64) ─► h₁ ─┐
User (5)       ─► Dense(64) ─► h₂ ─┼─► Attention ─► Weighted Sum ─► Dense ─► Output
Engagement (5) ─► Dense(64) ─► h₃ ─┘    (learned)
```

| Thành phần | Giải thích |
|------------|------------|
| **Dense(64)** | Projection layer - đưa mỗi modality về cùng dimension |
| **h₁, h₂, h₃** | Hidden representations của mỗi branch |
| **Attention** | Học weights dựa trên content: α = softmax(W × [h₁, h₂, h₃]) |
| **Weighted Sum** | output = α₁×h₁ + α₂×h₂ + α₃×h₃ |

**Attention mechanism chi tiết**:
```
        ┌─────────────────────────────────────────────┐
        │              ATTENTION LAYER                 │
        ├─────────────────────────────────────────────┤
        │                                              │
        │  Input: H = [h₁, h₂, h₃]  (shape: 3 × 64)   │
        │                                              │
        │  Step 1: Score = H × W + b                   │
        │          (W: 64×1, score: 3×1)               │
        │                                              │
        │  Step 2: α = softmax(Score)                  │
        │          α = [0.5, 0.3, 0.2] (tổng = 1)     │
        │                                              │
        │  Step 3: Output = Σ αᵢ × hᵢ                  │
        │          = 0.5×h₁ + 0.3×h₂ + 0.2×h₃         │
        │                                              │
        └─────────────────────────────────────────────┘
```

**Ưu điểm**: Tự học importance của từng modality, interpretable (xem weights)
**Nhược điểm**: Cần neural network, có thể overfit với data nhỏ
**Code concept (PyTorch)**:
```python
class AttentionFusion(nn.Module):
    def __init__(self, input_dims, hidden_dim=64):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)
    
    def forward(self, inputs):  # inputs = [text, user, engagement]
        # Project each modality
        hidden = [proj(x) for proj, x in zip(self.projections, inputs)]
        hidden = torch.stack(hidden, dim=1)  # (batch, 3, 64)
        
        # Attention weights
        scores = self.attention(hidden).squeeze(-1)  # (batch, 3)
        weights = F.softmax(scores, dim=1)  # (batch, 3)
        
        # Weighted sum
        output = (hidden * weights.unsqueeze(-1)).sum(dim=1)  # (batch, 64)
        
        return self.classifier(output), weights  # Return weights for interpretability
```

---

**5. GATED FUSION**

```
Text ────► Gate_T = σ(W_T × Text) ──► Text × Gate_T ──┐
User ────► Gate_U = σ(W_U × User) ──► User × Gate_U ──┼──► Concat ─► Dense ─► Output
Engage ──► Gate_E = σ(W_E × Eng) ───► Eng × Gate_E ───┘
```

| Thành phần | Giải thích |
|------------|------------|
| **σ (sigmoid)** | Hàm sigmoid, output trong [0, 1] |
| **Gate** | Quyết định "bao nhiêu %" information được pass qua |
| **×** | Element-wise multiplication |

**Khác với Attention**:
```
Attention: weights sum = 1 (softmax)
  → Nếu text quan trọng (0.8), user phải giảm (0.15), engagement giảm (0.05)
  → Branches "cạnh tranh" nhau

Gated: mỗi gate độc lập [0, 1]
  → Text gate = 0.9, User gate = 0.8, Engagement gate = 0.3
  → Có thể dùng nhiều hoặc ít TẤT CẢ branches
```

**Ưu điểm**: Linh hoạt hơn Attention, mỗi modality độc lập
**Nhược điểm**: Không normalize, có thể unstable
**Code concept (PyTorch)**:
```python
class GatedFusion(nn.Module):
    def __init__(self, input_dims, hidden_dim=64):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        self.gates = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        self.classifier = nn.Linear(hidden_dim * len(input_dims), 1)
    
    def forward(self, inputs):
        gated_outputs = []
        for proj, gate, x in zip(self.projections, self.gates, inputs):
            h = proj(x)  # (batch, 64)
            g = torch.sigmoid(gate(x))  # (batch, 64) - gate values 0-1
            gated_outputs.append(h * g)  # Element-wise gating
        
        concat = torch.cat(gated_outputs, dim=1)  # (batch, 64*3)
        return self.classifier(concat)
```

---

**6. MULTI-HEAD ATTENTION FUSION**

```
                        ┌─► Head 1 ─► weights₁ ─► output₁ ─┐
[h₁, h₂, h₃] ──────────┼─► Head 2 ─► weights₂ ─► output₂ ─┼─► Concat ─► Dense ─► Final
                        ├─► Head 3 ─► weights₃ ─► output₃ ─┤
                        └─► Head 4 ─► weights₄ ─► output₄ ─┘
```

| Thành phần | Giải thích |
|------------|------------|
| **Head** | Mỗi head là 1 attention mechanism riêng |
| **Nhiều heads** | Capture nhiều "aspects" khác nhau của relationship |
| **Concat** | Nối outputs từ tất cả heads |

**Ví dụ intuition**:
```
Head 1: Focus vào "credibility" → weight cao cho user features
Head 2: Focus vào "virality" → weight cao cho engagement features
Head 3: Focus vào "content" → weight cao cho text features
Head 4: Focus vào "controversy" → weight cao cho cmt_to_like ratio
```

**Ưu điểm**: Capture nhiều patterns, mạnh hơn single-head
**Nhược điểm**: Nhiều parameters hơn, cần tune num_heads
**Code concept (PyTorch)**:
```python
class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, input_dims, hidden_dim=64, num_heads=4):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_heads)
        ])
        self.classifier = nn.Linear(hidden_dim * num_heads, 1)
    
    def forward(self, inputs):
        hidden = [proj(x) for proj, x in zip(self.projections, inputs)]
        hidden = torch.stack(hidden, dim=1)  # (batch, 3, 64)
        
        head_outputs = []
        for head in self.heads:
            scores = head(hidden).squeeze(-1)  # (batch, 3)
            weights = F.softmax(scores, dim=1)  # (batch, 3)
            output = (hidden * weights.unsqueeze(-1)).sum(dim=1)  # (batch, 64)
            head_outputs.append(output)
        
        concat = torch.cat(head_outputs, dim=1)  # (batch, 64*num_heads)
        return self.classifier(concat)
```

---

**7. CROSS-ATTENTION FUSION**

```
        Text ◄─────────────────────────► Features (User + Engagement)
              │                        │
              │    CROSS-ATTENTION     │
              │                        │
              ▼                        ▼
        Text_enhanced           Features_enhanced
              │                        │
              └──────────┬─────────────┘
                         ▼
                    Concat/Add
                         ▼
                      Output
```

| Thành phần | Giải thích |
|------------|------------|
| **Query (Q)** | Từ modality A (ví dụ: Text) |
| **Key (K), Value (V)** | Từ modality B (ví dụ: Features) |
| **Cross-Attention** | A "attend" vào B: Attention(Q_A, K_B, V_B) |
| **Bidirectional** | Làm cả 2 chiều: Text→Features và Features→Text |

**Cách hoạt động chi tiết**:
```
Text attend to Features:
  Q = Text × W_Q           # Text làm query
  K = Features × W_K       # Features làm key
  V = Features × W_V       # Features làm value
  
  Attention_scores = Q × K.T / √d
  Attention_weights = softmax(Attention_scores)
  Text_enhanced = Attention_weights × V
  
  → Text được "enhanced" bởi thông tin từ Features
```

**Ví dụ intuition**:
```
Text có từ "BREAKING NEWS!!!" 
  → Cross-attention tăng weight cho `num_share` (vì breaking news thường được share nhiều)
  → Nếu `num_share` thấp bất thường → signal cho fake news

User có `num_fake` cao
  → Cross-attention focus vào emotional words trong text
  → Tìm pattern của fake news từ user này
```

**Ưu điểm**: Modalities interact với nhau, capture complex relationships
**Nhược điểm**: Phức tạp nhất, cần nhiều data, chậm hơn
**Code concept (PyTorch)**:
```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, text_dim, feat_dim, hidden_dim=64):
        super().__init__()
        # Text -> Features attention
        self.q_text = nn.Linear(text_dim, hidden_dim)
        self.k_feat = nn.Linear(feat_dim, hidden_dim)
        self.v_feat = nn.Linear(feat_dim, hidden_dim)
        
        # Features -> Text attention  
        self.q_feat = nn.Linear(feat_dim, hidden_dim)
        self.k_text = nn.Linear(text_dim, hidden_dim)
        self.v_text = nn.Linear(text_dim, hidden_dim)
        
        self.classifier = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, text, features):
        # Text attends to Features
        Q_t = self.q_text(text)      # (batch, hidden)
        K_f = self.k_feat(features)  # (batch, hidden)
        V_f = self.v_feat(features)  # (batch, hidden)
        
        attn_t = F.softmax(Q_t * K_f / np.sqrt(hidden_dim), dim=-1)
        text_enhanced = attn_t * V_f
        
        # Features attends to Text
        Q_f = self.q_feat(features)
        K_t = self.k_text(text)
        V_t = self.v_text(text)
        
        attn_f = F.softmax(Q_f * K_t / np.sqrt(hidden_dim), dim=-1)
        feat_enhanced = attn_f * V_t
        
        # Combine
        combined = torch.cat([text_enhanced, feat_enhanced], dim=1)
        return self.classifier(combined)
```

---

**8. HIERARCHICAL STACKING + ATTENTION FUSION**

```
┌────────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL STACKING + ATTENTION                │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  BRANCH 1: Text                                                     │
│  ├── XGBoost ──► P1a ─┐                                            │
│  ├── LightGBM ─► P1b ─┼─► Stack (LR) ──► P_text                    │
│  └── CatBoost ──► P1c ┘                                            │
│                                                                     │
│  BRANCH 2: User                                                     │
│  ├── XGBoost ──► P2a ─┐                                            │
│  ├── LightGBM ─► P2b ─┼─► Stack (LR) ──► P_user                    │
│  └── CatBoost ──► P2c ┘                                            │
│                                                                     │
│  BRANCH 3: Engagement                                               │
│  ├── XGBoost ──► P3a ─┐                                            │
│  ├── LightGBM ─► P3b ─┼─► Stack (LR) ──► P_engagement              │
│  └── CatBoost ──► P3c ┘                                            │
│                                                                     │
│  FINAL ATTENTION FUSION:                                            │
│  [P_text, P_user, P_engagement] ──► Attention ──► Final            │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

| Thành phần | Giải thích |
|------------|------------|
| **Branch Stacking** | Mỗi modality được stacking bởi 3 models |
| **P_text, P_user, P_engagement** | Predictions đã được stacked từ mỗi branch |
| **Attention Fusion** | Học weights cho 3 branch predictions |

**Ưu điểm**: 
- Kết hợp sức mạnh của cả Stacking (robust) và Attention (learned weights)
- Mỗi branch đã được optimize riêng trước khi fusion

**Nhược điểm**: 
- Phức tạp nhất, cần train nhiều models
- Tổng: 9 base models + 3 stack meta-learners + 1 attention = 13 models

**Code concept**:
```python
# Step 1: Branch Stacking
# Text branch
P1a = cross_val_predict(XGBClassifier(), text, y, cv=5, method='predict_proba')[:, 1]
P1b = cross_val_predict(LGBMClassifier(), text, y, cv=5, method='predict_proba')[:, 1]
P1c = cross_val_predict(CatBoostClassifier(), text, y, cv=5, method='predict_proba')[:, 1]
meta_text = LogisticRegression().fit(np.column_stack([P1a, P1b, P1c]), y)

# Tương tự cho User và Engagement branches...

# Step 2: Get branch predictions
P_text = meta_text.predict_proba(...)[:, 1]
P_user = meta_user.predict_proba(...)[:, 1]
P_engagement = meta_engagement.predict_proba(...)[:, 1]

# Step 3: Attention Fusion (Neural Network)
class FinalAttention(nn.Module):
    def __init__(self):
        self.attention = nn.Linear(3, 3)  # 3 branch predictions
        self.output = nn.Linear(3, 1)
    
    def forward(self, P_text, P_user, P_eng):
        inputs = torch.stack([P_text, P_user, P_eng], dim=1)  # (batch, 3)
        weights = F.softmax(self.attention(inputs), dim=1)
        weighted = inputs * weights
        return self.output(weighted.sum(dim=1))
```

---

##### 📊 SO SÁNH TẤT CẢ KIẾN TRÚC

| Kiến trúc | Complexity | Parameters | Interpretable | Tiềm năng | Đóng góp mới |
|-----------|------------|------------|---------------|-----------|--------------|
| Early Fusion | ⭐ | Thấp | ✅ Cao | ⭐⭐ | ❌ |
| Late Fusion | ⭐⭐ | Thấp | ✅ Cao | ⭐⭐⭐ | ❌ |
| Stacking | ⭐⭐ | Trung bình | ✅ Cao | ⭐⭐⭐ | ❌ |
| **Attention Fusion** | ⭐⭐⭐ | Trung bình | ✅ Cao (xem weights) | ⭐⭐⭐⭐ | ✅ |
| **Gated Fusion** | ⭐⭐⭐ | Trung bình | ⚠️ TB | ⭐⭐⭐⭐ | ✅ |
| **Multi-head Attention** | ⭐⭐⭐ | Cao | ✅ Cao | ⭐⭐⭐⭐ | ✅ |
| **Cross-Attention** | ⭐⭐⭐⭐ | Cao | ⚠️ TB | ⭐⭐⭐⭐⭐ | ✅ |
| **Hierarchical Stack+Attn** | ⭐⭐⭐⭐ | Rất cao | ✅ Cao | ⭐⭐⭐⭐⭐ | ✅ |

##### 📊 THỨ TỰ THỬ NGHIỆM KHUYẾN NGHỊ

| Thứ tự | Kiến trúc | Lý do |
|--------|-----------|-------|
| 1 | **Attention Fusion** | Default, cân bằng complexity/performance |
| 2 | **Gated Fusion** | Backup #1, nếu Attention không tốt |
| 3 | **Multi-head Attention** | Backup #2, mạnh hơn single-head |
| 4 | **Hierarchical Stack+Attn** | Backup #3, combine nhiều approaches |
| 5 | **Cross-Attention** | Last resort, phức tạp nhất |

#### 4.3 Tổng kết Models sẽ train

| # | Category | Model | Algorithm | Mô tả |
|---|----------|-------|-----------|-------|
| **BASELINES** |
| 1 | Single | Logistic Regression | LR | Baseline đơn giản nhất |
| 2 | Single | MLP | Neural | Neural baseline |
| **SINGLE BOOSTING** |
| 3 | Single | LightGBM | LightGBM | Nhanh |
| 4 | Single | XGBoost | XGBoost | Robust |
| 5 | Single | CatBoost | CatBoost | Auto-cat |
| **BASIC FUSION** |
| 6 | Fusion | Early Fusion | XGBoost | Concat → 1 model |
| 7 | Fusion | Late Fusion | XGBoost × 3 | Average predictions |
| 8 | Fusion | Stacking | XGBoost + LR | Meta-learner |
| **ADVANCED FUSION (PROPOSED)** |
| 9 | Fusion | **Attention Fusion** | MLP + Attention | Learned weights cho branches |
| 10 | Fusion | **Gated Fusion** | MLP + Gates | Gates điều khiển information |
| 11 | Fusion | **Multi-head Attention** | MLP + Multi-head | Capture nhiều aspects |
| 12 | Fusion | **Cross-Attention Fusion** | MLP + Cross-Attn | Modalities attend to each other |
| 13 | Fusion | **Hierarchical Stack+Attn** | Stacking + Attention | Best of both worlds |
| **DEEP LEARNING (Optional)** |
| 14 | DL | Fine-tuned PhoBERT | Transformer | SOTA text |

---

### PHASE 5: EVALUATION

#### 5.1 Metrics
| Metric | Mục đích | Priority |
|--------|----------|----------|
| **F1-Score** | Metric chính (cân bằng P và R) | 🔴 Cao |
| **Recall** | Bắt nhiều tin giả | 🔴 Cao |
| **Precision** | Ít oan tin thật | 🟡 TB |
| **Accuracy** | Baseline, dễ hiểu | 🟡 TB |
| **AUC-ROC** | Tổng thể, vẽ đồ thị | 🟡 TB |

#### 5.2 Validation Strategy
- [ ] Train/Val/Test split: 70/15/15 (stratified)
- [ ] K-Fold Cross-Validation: k=5

#### 5.3 Experiments Table (Template)
```
┌─────────────────────────────┬──────────┬───────────┬────────┬──────────┬─────────┐
│ Model                       │ Accuracy │ Precision │ Recall │ F1-Score │ AUC-ROC │
├─────────────────────────────┼──────────┼───────────┼────────┼──────────┼─────────┤
│ BASELINES                   │          │           │        │          │         │
│ Logistic Regression         │          │           │        │          │         │
│ MLP                         │          │           │        │          │         │
├─────────────────────────────┼──────────┼───────────┼────────┼──────────┼─────────┤
│ SINGLE BOOSTING             │          │           │        │          │         │
│ LightGBM                    │          │           │        │          │         │
│ XGBoost                     │          │           │        │          │         │
│ CatBoost                    │          │           │        │          │         │
├─────────────────────────────┼──────────┼───────────┼────────┼──────────┼─────────┤
│ FUSION                      │          │           │        │          │         │
│ Early Fusion (XGBoost)      │          │           │        │          │         │
│ Late Fusion (XGBoost × 3)   │          │           │        │          │         │
│ Stacking (XGBoost + LR)     │          │           │        │          │         │
│ Attention Fusion (PROPOSED) │          │           │        │          │         │
├─────────────────────────────┼──────────┼───────────┼────────┼──────────┼─────────┤
│ DEEP LEARNING (Optional)    │          │           │        │          │         │
│ Fine-tuned PhoBERT          │          │           │        │          │         │
└─────────────────────────────┴──────────┴───────────┴────────┴──────────┴─────────┘
```

---

### PHASE 6: ANALYSIS (Chỉ làm với BEST model)

#### 5.1 Ablation Study
Chứng minh từng component có đóng góp:
```
┌─────────────────────────────────┬────────┐
│ Configuration                   │   F1   │
├─────────────────────────────────┼────────┤
│ Full Model (proposed)           │        │
│ − Bỏ Attention (dùng concat)    │        │
│ − Bỏ User Features              │        │
│ − Bỏ Engagement Features        │        │
│ − Chỉ Text                      │        │
│ − Chỉ Features                  │        │
└─────────────────────────────────┴────────┘
```

#### 5.2 Error Analysis
- [ ] Phân loại lỗi: False Positive vs False Negative
- [ ] Tìm pattern trong lỗi: topic, text length, confidence level
- [ ] Case studies: 3-5 ví dụ điển hình

#### 5.3 Explainability (XAI)
- [ ] Feature Importance (SHAP hoặc Tree-based)
- [ ] Giải thích prediction bằng lời

#### 5.4 Statistical Significance Test
- [ ] Paired t-test giữa best model và runner-up
- [ ] Báo cáo p-value

---

### PHASE 7: COMPARISON & CONTRIBUTION

#### 6.1 So sánh với SOTA
- [ ] Tìm papers về Vietnamese Fake News Detection
- [ ] So sánh metrics trên cùng dataset (nếu có)

#### 6.2 Contributions (Đóng góp của nghiên cứu)
1. Đề xuất kiến trúc Attention Fusion cho Fake News Detection tiếng Việt
2. Phân tích thống kê toàn diện các features với Cohen's d, KS statistic
3. Ablation study chứng minh đóng góp của từng component
4. Error analysis chỉ ra các trường hợp khó phát hiện
5. Đạt F1-score XX%, vượt trội so với SOTA

---

## ⏰ CHECKLIST THỰC HIỆN

### Phase 1: Data & Features
- [ ] Trích xuất tất cả features
- [ ] Tính Cohen's d, KS, T-test cho từng feature
- [ ] Vẽ biểu đồ (boxplot, KDE, heatmap)
- [ ] Chọn features cuối cùng

### Phase 2: Text Processing
- [ ] Text preprocessing
- [ ] TF-IDF embedding
- [ ] PhoBERT/BERT embedding

### Phase 3: Dimensionality Reduction
- [ ] Truncated SVD cho TF-IDF (→ 100-300 chiều)
- [ ] PCA cho BERT embeddings (optional)
- [ ] Thử nghiệm số chiều tối ưu

### Phase 4: Modeling
- [ ] **Baselines**: Logistic Regression, MLP
- [ ] **Single Boosting**: LightGBM, XGBoost, CatBoost
- [ ] **Fusion**: Early Fusion, Late Fusion, Stacking
- [ ] **Proposed**: Attention Fusion
- [ ] **Deep Learning** (optional): Fine-tuned PhoBERT

### Phase 5: Evaluation
- [ ] Tính metrics cho tất cả models (Acc, P, R, F1, AUC)
- [ ] K-fold Cross-Validation (k=5)
- [ ] Chọn best model

### Phase 6: Analysis (chỉ best model)
- [ ] Ablation Study
- [ ] Error Analysis
- [ ] Explainability (SHAP)
- [ ] Statistical significance test

### Phase 7: Finalization
- [ ] So sánh với SOTA papers
- [ ] Viết contributions
- [ ] Hoàn thiện báo cáo

---

## 📚 REFERENCES

### Vietnamese NLP Models
- PhoBERT: https://github.com/VinAIResearch/PhoBERT
- ViSoBERT: https://github.com/uitnlp/ViSoBERT
- BARTpho: https://github.com/VinAIResearch/BARTpho
- ViT5: https://github.com/vietai/ViT5

### Multilingual Models
- XLM-RoBERTa: https://huggingface.co/xlm-roberta-base
- mBERT: https://huggingface.co/bert-base-multilingual-cased
- InfoXLM: https://github.com/microsoft/unilm/tree/master/infoxlm
- RemBERT: https://huggingface.co/google/rembert

### Sentence Transformers & Retrieval
- E5 (Multilingual): https://huggingface.co/intfloat/multilingual-e5-large
- BGE-M3: https://huggingface.co/BAAI/bge-m3
- GTE-multilingual: https://huggingface.co/Alibaba-NLP/gte-multilingual-base
- Jina-embeddings-v3: https://huggingface.co/jinaai/jina-embeddings-v3

### Latest SOTA (2024-2025)
- NV-Embed-v2: https://huggingface.co/nvidia/NV-Embed-v2
- GritLM: https://github.com/ContextualAI/gritlm
- Stella: https://huggingface.co/dunzhang/stella_en_1.5B_v5
- SFR-Embedding: https://huggingface.co/Salesforce/SFR-Embedding-2_R
- LLM2Vec: https://github.com/McGill-NLP/llm2vec
- Nomic-embed: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- Arctic-embed: https://huggingface.co/Snowflake/snowflake-arctic-embed-l
- mxbai-embed: https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1

### Benchmarks & Leaderboards
- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- Sentence Transformers: https://www.sbert.net/

### Explainability
- SHAP: https://github.com/slundberg/shap

---

## 🛠️ INSTALLATION

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 👥 AUTHORS

- [Your Name]

## 📄 LICENSE

MIT License