from flask import Flask, jsonify, request
from flask_cors import CORS

# 初始化 Flask app
app = Flask(__name__)
# 设置 CORS，允许来自前端（例如 http://localhost:5173）的跨域请求
CORS(app)

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    接收来自前端的RNA序列，未来将调用分析逻辑，目前返回模拟数据。
    """
    # 从请求的JSON body中获取数据
    data = request.get_json()
    sequence = ""
    if data:
        sequence = data.get('sequence')
        # 打印一下，确认收到了前端的值
        print(f"Received sequence from frontend: {sequence}")

    # 模拟的返回数据
    mock_data = {
        "classification": {
            "name": "RNA Sequence",
            "isPredicted": True,
            "children": [
                {"name": "Group A", "isPredicted": True, "children": [{"name": "Mod 1", "isPredicted": True}]},
                {"name": "Group B", "isPredicted": False, "children": [{"name": "Mod 4", "isPredicted": False}]},
            ]
        },
        "attention": {
            "sequence": sequence if sequence else "AUGCCGUACGAUCGACGAUCGUACGUACGUACGUAUCGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACG",
            "weights": [
                {"index": 3, "type": "m6A", "score": 0.98},
                {"index": 15, "type": "m5C", "score": 0.95},
                {"index": 33, "type": "ac4C", "score": 0.91},
                {"index": 45, "type": "m1A", "score": 0.89},
                {"index": 60, "type": "m6A", "score": 0.88},
            ]
        },
        "gcn": {
            "nodes": [
                {"id": "A1", "label": "位置1: A (腺嘌呤)", "data": {"index": 1, "type": "A", "name": "腺嘌呤"}},
                {"id": "U2", "label": "位置2: U (尿嘧啶)", "data": {"index": 2, "type": "U", "name": "尿嘧啶"}},
                {"id": "G3", "label": "位置3: G (鸟嘌呤)", "data": {"index": 3, "type": "G", "name": "鸟嘌呤"}},
                {"id": "C4", "label": "位置4: C (胞嘧啶)", "data": {"index": 4, "type": "C", "name": "胞嘧啶"}},
                {"id": "A5", "label": "位置5: A (腺嘌呤)", "data": {"index": 5, "type": "A", "name": "腺嘌呤"}},
                {"id": "U6", "label": "位置6: U (尿嘧啶)", "data": {"index": 6, "type": "U", "name": "尿嘧啶"}},
                {"id": "C7", "label": "位置7: C (胞嘧啶)", "data": {"index": 7, "type": "C", "name": "胞嘧啶"}},
                {"id": "G8", "label": "位置8: G (鸟嘌呤)", "data": {"index": 8, "type": "G", "name": "鸟嘌呤"}}
            ],
            "edges": [
                {"source": "A1", "target": "U2"},
                {"source": "U2", "target": "G3"},
                {"source": "G3", "target": "C4"},
                {"source": "C4", "target": "A5"},
                {"source": "A5", "target": "U6"},
                {"source": "U6", "target": "C7"},
                {"source": "C7", "target": "G8"},
                {"source": "A1", "target": "G3"},
                {"source": "U2", "target": "C4"},
                {"source": "G3", "target": "A5"}
            ]
        }
    }

    return jsonify(mock_data)

if __name__ == '__main__':
    # 运行服务器在 5000 端口
    app.run(debug=True, port=5000)
