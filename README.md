# Vietnamese-Fake-News-Detection
Xây dựng model dự đoán thông tin giả mạng xã hội tiếng Việt

## 📋 Thông tin dự án
- **Python version**: 3.12.0
- **Virtual environment**: .venv/
- **Dataset**: 4736 bài viết (Real: 3929, Fake: 807)
- **SOTA to beat**: AUC 96.47% (Public), 95.21% (Private)

## 🎯 MỤC TIÊU NGHIÊN CỨU

1. Với bài toán này việc quan trọng nhất là biểu diễn text thành vecto tốt nhất có thể. Vì vậy việc đầu tiên và quan trọng nhất sẽ là lựa chọn embeding phù hợp dựa trên việc cố định model đằng sau là logistic regression
* TF-IDF

* Word2Vec
* FastText
* GloVe

* PhoBERT-v2
* PhoBERT-v2 finetuned

2. Sau khi chọn được embedding tốt nhất sẽ trích xuất thêm các đặc trưng bên ngoài để làm tăng lượng thông tin cho mô hình, sau đó sẽ lựa chọn model tốt nhất để dự đoán.

3. Tiếp theo là lựa chọn model phù hợp khi đã có vecto biểu diễn hợp lí, ta sẽ thử nghiệm với các mô hình khác nhau để tìm ra mô hình tốt nhất cho bài toán này dựa trên valid set.

* Machine learning: Logistic Regression, LightGBM, multilayer perceptron (MLP), essemble
* Deep learning: CNN 1D, LSTM, Bert encoder

4. Sau khi chọn được model tốt nhất chúng ta sẽ phân tích kết quả dự đoán để tìm ra những điểm mạnh và điểm yếu của mô hình, từ đó có thể đưa ra những cải tiến cho mô hình trong tương lai.

5. Cuối cùng là đánh giá kết quả trên test set và các dataset khác để xem mô hình hoạt động như thế nào trên dữ liệu chưa từng thấy trước đó, từ đó có thể so sánh với những nghiên cứu về bài toán này và rút ra những bài học kinh nghiệm cho việc xây dựng mô hình trong tương lai.



