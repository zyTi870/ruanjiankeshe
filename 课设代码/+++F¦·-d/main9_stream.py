#!/usr/bin/env python3
# main9_stream.py

import os
# Suppress OpenCV warnings (best effort; may be read at import time on some builds)
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np
from datetime import datetime
import threading
import queue
import time
import argparse


class CircleDetector:
    def __init__(
        self,
        output_dir="captures",
        camera_id="/dev/video3",
        frame_width=1920,
        frame_height=1080,
        processing_scale=0.3,
        process_every_n_frames=2,
        min_contour_area=500,
        max_aspect_ratio=1.4,
    ):
        """
        camera_id 支持两种形式：
        - 数字/数字字符串：例如 0 / "0"（按索引打开）
        - 设备路径：例如 "/dev/video3"（强制打开对应设备，更稳定）
        """
        self.PROCESS_EVERY_N_FRAMES = process_every_n_frames
        self.processing_scale = processing_scale
        self.frame_counter = 0
        self.last_known_detection = None
        self.snapshot_count = 0
        self.min_contour_area = min_contour_area
        self.max_aspect_ratio = max_aspect_ratio

        # ---- 关键改动：支持 /dev/videoX，且固定使用 V4L2 后端 ----
        self.cap = self._open_camera(camera_id)
        if self.cap is None or (hasattr(self.cap, "isOpened") and not self.cap.isOpened()):
            print(f"[ERROR] Could not open camera: {camera_id}")
            self.stopped = True
            return

        # 尝试设置分辨率（实际以驱动返回为准）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[INFO] Attempting to set resolution to: {frame_width}x{frame_height}")
        print(f"[INFO] Actual camera resolution: {int(actual_width)}x{int(actual_height)}")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[INFO] Snapshots will be saved in '{self.output_dir}/' directory.")

        self.stopped = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.reader_thread.start()
        print("[INFO] Camera reader thread started.")

        # server 模式：缓存最新“已叠加”帧
        self._latest_annotated = None
        self._latest_lock = threading.Lock()

    def _open_camera(self, camera_id):
        """
        更稳的打开方式：
        - /dev/videoX 用路径打开（避免索引映射不一致）
        - 数字/字符串数字用索引打开
        """
        try:
            if isinstance(camera_id, str):
                camera_id = camera_id.strip()
                if camera_id.startswith("/dev/video"):
                    return cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
                # 允许传 "3"
                if camera_id.isdigit():
                    return cv2.VideoCapture(int(camera_id), cv2.CAP_V4L2)
            # 允许传 int
            return cv2.VideoCapture(int(camera_id), cv2.CAP_V4L2)
        except Exception as e:
            print(f"[ERROR] open_camera failed: {e}")
            return None

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break

            # 永远只保留最新帧
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def process_frame_color_segmentation(self, frame):
        if frame is None or frame.size == 0:
            return None

        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=self.processing_scale,
            fy=self.processing_scale,
            interpolation=cv2.INTER_AREA,
        )
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)

        # 紫色范围（你调好的）
        lower_bound = np.array([125, 40, 40])
        upper_bound = np.array([155, 255, 255])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if cv2.__version__.startswith("4"):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_detection_info = None
        max_circle_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue
            if len(cnt) < 5:
                continue

            ellipse = cv2.fitEllipse(cnt)
            (center, axes, _angle) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            if minor_axis == 0:
                continue

            aspect_ratio = major_axis / minor_axis
            if aspect_ratio < self.max_aspect_ratio:
                if area > max_circle_area:
                    max_circle_area = area
                    center_x = int(center[0] / self.processing_scale)
                    center_y = int(center[1] / self.processing_scale)
                    radius = int((major_axis + minor_axis) / 4 / self.processing_scale)
                    scaled_area = area / (self.processing_scale**2)

                    best_detection_info = {
                        "center": (center_x, center_y),
                        "radius": radius,
                        "area": int(scaled_area),
                        "aspect_ratio": float(aspect_ratio),
                    }

        return best_detection_info

    def annotate_frame(self, frame):
        """对一帧做：按节流策略更新检测 + 叠加可视化信息，返回叠加后的帧(BGR)"""
        display_frame = frame.copy()

        if self.frame_counter % self.PROCESS_EVERY_N_FRAMES == 0:
            new_detection = self.process_frame_color_segmentation(frame)
            if new_detection is not None:
                self.last_known_detection = new_detection

        self.frame_counter += 1

        if self.last_known_detection is not None:
            center = self.last_known_detection["center"]
            radius = self.last_known_detection["radius"]
            cv2.circle(display_frame, center, radius, (0, 255, 0), 3)
            cv2.circle(display_frame, center, 5, (0, 0, 255), -1)

            info_y = 30
            info_font = cv2.FONT_HERSHEY_SIMPLEX
            info_color = (0, 255, 255)
            font_scale = 0.7
            font_thickness = 2

            ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(
                display_frame,
                f"Timestamp: {ts_str}",
                (15, info_y),
                info_font,
                font_scale,
                info_color,
                font_thickness,
            )
            info_y += 30

            geo = self.last_known_detection
            cv2.putText(
                display_frame,
                f"Center (px): {geo['center']}",
                (15, info_y),
                info_font,
                font_scale,
                info_color,
                font_thickness,
            )
            info_y += 30
            cv2.putText(
                display_frame,
                f"Radius (px): {geo['radius']}",
                (15, info_y),
                info_font,
                font_scale,
                info_color,
                font_thickness,
            )
            info_y += 30
            cv2.putText(
                display_frame,
                f"Area (px^2): {geo['area']}",
                (15, info_y),
                info_font,
                font_scale,
                info_color,
                font_thickness,
            )
            info_y += 30
            cv2.putText(
                display_frame,
                f"Aspect Ratio: {geo['aspect_ratio']:.3f}",
                (15, info_y),
                info_font,
                font_scale,
                info_color,
                font_thickness,
            )
            info_y += 30
            cv2.putText(
                display_frame,
                f"Snapshot Count: {self.snapshot_count}",
                (15, info_y),
                info_font,
                font_scale,
                info_color,
                font_thickness,
            )

        return display_frame

    def run_local(self):
        """EB2 本地显示模式"""
        if self.stopped:
            return
        print("[INFO] Local preview... Press 's' to save, 'q' to quit.")

        while not self.stopped:
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.005)
                if not self.reader_thread.is_alive():
                    break
                continue

            display_frame = self.annotate_frame(frame)

            h, w = display_frame.shape[:2]
            preview_h, preview_w = int(h * 0.7), int(w * 0.7)
            cv2.imshow("Camera Feed", cv2.resize(display_frame, (preview_w, preview_h)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.stop()
                break
            elif key == ord("s"):
                self.save_snapshot(display_frame)

        self.stop()
        print("[INFO] Program exited.")

    def run_server_worker(self):
        """server 模式：不断生成最新叠加帧，供 /video_feed 读取"""
        while not self.stopped:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            annotated = self.annotate_frame(frame)
            with self._latest_lock:
                self._latest_annotated = annotated

    def get_latest_annotated(self):
        with self._latest_lock:
            return None if self._latest_annotated is None else self._latest_annotated.copy()

    def get_status(self):
        det = self.last_known_detection
        return {
            "snapshot_count": int(self.snapshot_count),
            "frame_counter": int(self.frame_counter),
            "detection": det,
        }

    def save_snapshot(self, frame_to_save):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.output_dir, f"capture_{timestamp}.jpg")
        cv2.imwrite(filename, frame_to_save)
        self.snapshot_count += 1
        print(f"[INFO] Snapshot saved: {filename} (Total: {self.snapshot_count})")
        return filename

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        if hasattr(self, "reader_thread") and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2)
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


def run_flask_server(detector: CircleDetector, host="0.0.0.0", port=5000, jpeg_quality=80):
    from flask import Flask, Response, jsonify

    if getattr(detector, "stopped", False):
        print("[ERROR] Detector init failed; server will not start.")
        return

    app = Flask(__name__)

    def mjpeg_gen():
        while not detector.stopped:
            frame = detector.get_latest_annotated()
            if frame is None:
                time.sleep(0.02)
                continue
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            )
            if not ok:
                continue
            jpg = buf.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )

    @app.get("/health")
    def health():
        return jsonify({"ok": True})

    @app.get("/status")
    def status():
        return jsonify(detector.get_status())

    @app.get("/video_feed")
    def video_feed():
        return Response(mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # 后台处理线程：持续更新 detector._latest_annotated
    t = threading.Thread(target=detector.run_server_worker, daemon=True)
    t.start()

    print(f"[INFO] MJPEG: http://{host}:{port}/video_feed")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "server"], default="server")
    # 关键改动：type=str，默认直接用 /dev/video3
    ap.add_argument("--camera_id", type=str, default="/dev/video3")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    detector = CircleDetector(
        camera_id=args.camera_id,
        frame_width=1920,
        frame_height=1080,
        process_every_n_frames=2,
        processing_scale=0.3,
        min_contour_area=500,
        max_aspect_ratio=1.4,
    )

    if args.mode == "local":
        detector.run_local()
    else:
        run_flask_server(detector, host="0.0.0.0", port=args.port, jpeg_quality=80)
