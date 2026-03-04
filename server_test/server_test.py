from flask import Flask, request, jsonify
import random
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận dữ liệu từ extension
    data = request.json
    print("Received data from extension:", data)

    # Tạo kết quả ngẫu nhiên (true/false)
    result = {
        "label": random.choice([0, 1]),  # 0: True news, 1: Fake news
        "confidence": round(random.uniform(0.5, 1.0), 2)  # Độ tin cậy ngẫu nhiên
    }

    print("Sending result to extension:", result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)