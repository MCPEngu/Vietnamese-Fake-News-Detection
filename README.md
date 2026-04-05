# Vietnamese-Fake-News-Detection
Xây dựng model dự đoán thông tin giả mạng xã hội tiếng Việt

## 📋 Thông tin dự án
- **Python version**: 3.12.0
- **Virtual environment**: `.venv/`
- **Dataset**: 4736 bài viết (Real: 3929, Fake: 807)
- **SOTA to beat**: AUC 96.47% (Public), 95.21% (Private)

---

## 🎯 MỤC TIÊU NGHIÊN CỨU

Với bài toán này việc quan trọng nhất là biểu diễn text thành vecto tốt nhất có thể.

1. Việc đầu tiên sẽ lựa chọn embeding dựa trên việc cố định model đằng sau là logistic regression
- **TF-IDF**: 

- **Word2Vec**: 
- **FastText**: 
- **GloVe**: 

- **PhoBERT-v2**: 
- **PhoBERT-v2 finetuned**:

- **Chat GPT**: text-embedding-3-small

2. Sau khi chọn được embedding tốt nhất sẽ trích xuất thêm các đặc trưng bên ngoài để làm tăng lượng thông tin cho mô hình, sau đó sẽ lựa chọn model tốt nhất để dự đoán.

3. Cuối cùng là lựa chọn model phù hợp khi đã có vecto biểu diễn hợp lí.

* Machine learning: Logistic Regression, LightGBM, multilayer perceptron (MLP), essemble
* Deep learning: CNN 1D, LSTM, Bert encoder


