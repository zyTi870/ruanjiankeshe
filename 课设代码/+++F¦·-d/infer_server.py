# infer_server.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import io
import json
import time
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify

# ========== 配置 ==========
MODEL_PATH = os.environ.get("MODEL_PATH", "best.onnx")  # 默认同目录 best.onnx
HOST = "0.0.0.0"
PORT = int(os.environ.get("INFER_PORT", "5001"))

app = Flask(__name__)

# ========== 推理器 ==========
class ARMClassifierONNXRT:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        import onnx
        import onnxruntime as ort

        self.model_path = model_path
        self.model = onnx.load(model_path)
        onnx.checker.check_model(self.model)

        # input shape
        input_tensor = self.model.graph.input[0]
        dims = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        # NCHW => dims = [1,3,H,W]
        self.input_h, self.input_w = int(dims[2]), int(dims[3])
        self.input_name = input_tensor.name

        # 建议只创建一次 Session（不要每次请求创建）
        self.sess = ort.InferenceSession(model_path)

    def preprocess_bgr(self, img_bgr: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        img_norm = img_resized.astype(np.float32) / 255.0
        img_chw = img_norm.transpose(2, 0, 1)  # HWC->CHW
        img_input = np.expand_dims(img_chw, axis=0).astype(np.float32)  # 1,3,H,W
        return img_input

    def infer_bgr(self, img_bgr: np.ndarray):
        x = self.preprocess_bgr(img_bgr)
        t0 = time.time()
        out = self.sess.run(None, {self.input_name: x})
        latency_ms = (time.time() - t0) * 1000.0

        logits = out[0][0]  # (num_classes,)
        # softmax
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        class_id = int(np.argmax(probs))
        score = float(probs[class_id])
        return class_id, score, latency_ms


# ========== 全局加载 ==========
try:
    clf = ARMClassifierONNXRT(MODEL_PATH)
    print(f"[OK] Model loaded: {MODEL_PATH}, input={clf.input_h}x{clf.input_w}")
except Exception as e:
    clf = None
    print(f"[ERROR] Failed to load model: {e}")


def decode_image_file(file_storage):
    """从 Flask 上传的文件对象解码为 BGR ndarray"""
    data = file_storage.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def encode_png_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@app.get("/health")
def health():
    return jsonify({"ok": True, "model_loaded": clf is not None, "model_path": MODEL_PATH})


@app.post("/infer_batch")
def infer_batch():
    """
    接收 PC 端 multipart 上传的 4 张图片：
      files: images=... (可以多个同名 images)
      form: levels=json list, imgsz=..., conf_thres=...
    返回：
      {"results":[{"level":..., "pred_label":..., "pred_conf":..., "pred_concentration":..., "heatmap_png_b64":...}, ...]}
    """
    if clf is None:
        return jsonify({"error": "model not loaded"}), 500

    images = request.files.getlist("images")
    if not images:
        return jsonify({"error": "no images uploaded, use field name 'images'"}), 400

    # levels: PC 端传的是 json 字符串
    levels_raw = request.form.get("levels", "[]")
    try:
        levels = json.loads(levels_raw)
    except Exception:
        levels = []

    # 如果 levels 缺失/长度不一致，就用 0..N-1 兜底
    if len(levels) != len(images):
        levels = list(range(len(images)))

    results = []
    for idx, f in enumerate(images):
        img = decode_image_file(f)
        if img is None:
            results.append({
                "level": float(levels[idx]),
                "pred_label": "decode_failed",
                "pred_conf": 0.0,
                "pred_concentration": 0.0,
                "heatmap_png_b64": ""
            })
            continue

        class_id, score, latency_ms = clf.infer_bgr(img)

        # 这里的 label 映射你后面按实际类别改（示例）
        # 比如 4 类浓度：0/33/66/100
        label_map = {
            0: "0%",
            1: "33%",
            2: "66%",
            3: "100%"
        }
        pred_label = label_map.get(class_id, f"class_{class_id}")

        # 如果你希望输出“预测浓度(%)”，这里给一个简单映射（按你的真实模型改）
        conc_map = {0: 0.0, 1: 33.0, 2: 66.0, 3: 100.0}
        pred_conc = float(conc_map.get(class_id, 0.0))

        # 你后续要 Grad-CAM：需要模型支持并额外实现；目前先返回空热力图
        results.append({
            "level": float(levels[idx]),
            "pred_label": pred_label,
            "pred_conf": float(score),
            "pred_concentration": pred_conc,
            "latency_ms": float(latency_ms),
            "heatmap_png_b64": ""  # 暂空：后续你接入 Grad-CAM 再填
        })

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, threaded=True)
