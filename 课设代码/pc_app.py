# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import altair as alt
from datetime import datetime
import os
import requests
import base64
import json
import re

# =========================================================
# 1) 类别映射：1~5
# =========================================================
CLASS_NAME_MAP = {
    1: "对照",
    2: "顺铂",
    3: "左旋肉碱",
    4: "对乙酰氨基酚",
    5: "苯比a",
}
INV_CLASS_NAME_MAP = {v: k for k, v in CLASS_NAME_MAP.items()}


def _first_existing(d: dict, keys, default=None):
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return default


def extract_class_id(item: dict):
    """
    从后端 item 中提取类别ID（优先返回 1~5）。
    兼容：
      - pred_label / pred_class / class_id / cls / label / top1 / pred_id
      - int / float / '1' / '1.0' / 'class_1' / '类别1'
      - 概率向量 list/ndarray -> argmax + 1
      - 若后端用 0~4 编码，会自动 +1 映射到 1~5
      - 若后端直接返回中文类名，也能反查为 ID
    """
    raw = _first_existing(
        item,
        keys=["pred_label", "pred_class", "class_id", "cls", "label", "top1", "pred_id"],
        default=None,
    )
    if raw is None:
        return None

    # 1) 概率向量 / one-hot
    if isinstance(raw, (list, tuple, np.ndarray)):
        arr = np.array(raw, dtype=np.float32).reshape(-1)
        if arr.size >= 2:
            k = int(np.argmax(arr))
            # 多数模型输出 0~4
            if (k + 1) in CLASS_NAME_MAP:
                return k + 1
            if k in CLASS_NAME_MAP:
                return k
        return None

    # 2) 直接中文类名
    if isinstance(raw, str):
        s = raw.strip()
        if s in INV_CLASS_NAME_MAP:
            return INV_CLASS_NAME_MAP[s]
    else:
        s = str(raw).strip()

    # 3) 数字解析：兼容 "1.0"
    try:
        k = int(float(s))
    except Exception:
        # 4) 从字符串里抽数字：class_1 / 类别1
        m = re.search(r"(\d+)", s)
        if not m:
            return None
        k = int(m.group(1))

    # 5) 兼容 0~4 -> 1~5
    if k in CLASS_NAME_MAP:
        return k
    if (k + 1) in CLASS_NAME_MAP:
        return k + 1

    return None


def class_name_from_item(item: dict) -> str:
    cid = extract_class_id(item)
    if cid is None:
        # fallback：直接给出后端字段
        raw = _first_existing(item, ["pred_label", "pred_class", "class_id", "cls", "label"], default="N/A")
        return str(raw)
    return CLASS_NAME_MAP.get(cid, f"未知({cid})")


# =========================================================
# 2) Utils
# =========================================================
def fetch_one_frame_from_mjpeg(url: str, timeout: float = 6.0, max_bytes: int = 2_000_000):
    """从 MJPEG 流中抓取一帧 JPEG 并解码为 BGR。"""
    headers = {"Cache-Control": "no-cache", "Pragma": "no-cache"}
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
        r.raise_for_status()
        data = bytearray()
        SOI, EOI = b"\xff\xd8", b"\xff\xd9"
        for chunk in r.iter_content(chunk_size=4096):
            if not chunk:
                continue
            data.extend(chunk)
            if len(data) > max_bytes:
                data = data[len(data) // 2 :]
            a = data.find(SOI)
            b = data.find(EOI)
            if a != -1 and b != -1 and b > a:
                jpg = bytes(data[a : b + 2])
                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return frame
    return None


def b64_to_bgr(b64str: str) -> np.ndarray:
    """base64(PNG/JPG bytes) -> BGR image，兼容 data:image/png;base64,..."""
    if not b64str:
        return None
    s = str(b64str).strip()
    if s.startswith("data:image"):
        s = s.split(",", 1)[1].strip()
    raw = base64.b64decode(s.encode("utf-8"))
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def parse_levels(s: str):
    vals = [x.strip() for x in s.split(",") if x.strip()]
    levels = [float(v) for v in vals]
    if len(levels) != 4:
        raise ValueError("梯度必须恰好 4 个，例如：0,33,66,100")
    return levels


def ensure_odd(k: int) -> int:
    if k <= 1:
        return 1
    return k if (k % 2 == 1) else k + 1


def gen_fake_gradcam_overlay(
    img_bgr: np.ndarray,
    seed: int = 0,
    alpha: float = 0.45,
    blur_ksize: int = 51,
    colormap: int = cv2.COLORMAP_JET,
):
    """前端伪热力图：随机热图+高斯模糊+叠加"""
    if img_bgr is None or img_bgr.size == 0:
        return None, None

    h, w = img_bgr.shape[:2]
    rng = np.random.default_rng(int(seed))
    heat = rng.random((h, w), dtype=np.float32)

    k = ensure_odd(int(blur_ksize))
    k = min(k, (min(h, w) // 2) * 2 - 1) if min(h, w) >= 5 else 1
    k = ensure_odd(max(k, 1))
    if k > 1:
        heat = cv2.GaussianBlur(heat, (k, k), 0)

    heat_u8 = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, colormap)

    a = float(np.clip(alpha, 0.0, 1.0))
    overlay = cv2.addWeighted(img_bgr, 1.0 - a, heat_color, a, 0.0)
    return heat_color, overlay


def safe_imread(path: str):
    if not path or not os.path.exists(path):
        return None
    return cv2.imread(path)


def bgr_to_rgb(img_bgr: np.ndarray):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# =========================================================
# 3) Streamlit config + CSS
# =========================================================
st.set_page_config(page_title="智能显微镜菌种识别系统", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
.main { background-color: #f5f7f9; }
.stButton>button { width: 100%; border-radius: 6px; height: 3em; background-color: #007bff; color: white; }
.img-container { border: 2px solid #ddd; border-radius: 10px; padding: 10px; background: white; }
.small-note { color: #666; font-size: 12px; }
.badge-ok { color: #1f7a1f; font-weight: 600; }
.badge-warn { color: #b26a00; font-weight: 600; }

.card {
  background:#ffffff;
  border:1px solid #e6e6e6;
  border-radius:14px;
  padding:14px;
}
.card-title {
  font-size:13px;color:#666;margin-bottom:8px;
}
.card-big {
  font-size:32px;font-weight:900;line-height:1.1;
}
.card-name {
  margin-top:10px;
  font-size:18px;font-weight:800;line-height:1.25;
  word-break:break-word;white-space:normal;
}
.legend {
  background:#ffffff;border:1px solid #e6e6e6;border-radius:14px;padding:12px;
  font-size:14px;line-height:1.6;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 4) Session State
# =========================================================
if "running" not in st.session_state:
    st.session_state.running = False
if "capture_session_dir" not in st.session_state:
    st.session_state.capture_session_dir = None
if "captured" not in st.session_state:
    st.session_state.captured = {}  # {level: path}
if "infer_results" not in st.session_state:
    st.session_state.infer_results = None
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "fake_cam_seed" not in st.session_state:
    st.session_state.fake_cam_seed = 12345


# =========================================================
# 5) Sidebar
# =========================================================
st.sidebar.title("实验控制面板")

with st.sidebar.expander("网络连接", expanded=True):
    eb2_ip = st.text_input("EB2 IP 地址", value="192.168.3.1")
    stream_port = st.number_input("视频流端口", 1, 65535, 5000, 1)
    infer_port = st.number_input("推理服务端口", 1, 65535, 5001, 1)

    video_url = f"http://{eb2_ip}:{stream_port}/video_feed"
    status_url = f"http://{eb2_ip}:{stream_port}/status"
    infer_url = f"http://{eb2_ip}:{infer_port}/infer_batch"

    st.markdown(
        f"<div class='small-note'>视频流：{video_url}<br/>推理API：{infer_url}</div>",
        unsafe_allow_html=True,
    )

with st.sidebar.expander("模型/实验配置", expanded=True):
    levels_str = st.text_input("浓度梯度(%)：逗号分隔（必须4个）", value="0,33,66,100")
    st.markdown(
        "<div class='small-note'>类别编号：1=对照，2=顺铂，3=左旋肉碱，4=对乙酰氨基酚，5=苯比a</div>",
        unsafe_allow_html=True,
    )

with st.sidebar.expander("前端伪热力图（当后端不返回热力图时）", expanded=True):
    use_fake_cam = st.checkbox("启用前端随机热力图填充", value=True)
    cam_alpha = st.slider("热力图叠加透明度 alpha", 0.0, 1.0, 0.45, 0.01)
    cam_blur = st.slider("热力图平滑程度（高斯核大小）", 1, 151, 51, 2)
    cam_seed = st.number_input("随机种子（固定热力图以便复现实验）", value=int(st.session_state.fake_cam_seed), step=1)
    st.session_state.fake_cam_seed = int(cam_seed)

st.sidebar.markdown("---")
start_btn = st.sidebar.button("开启实时预览")
stop_btn = st.sidebar.button("停止预览")
if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

try:
    levels = parse_levels(levels_str)
except Exception as e:
    st.sidebar.error(f"梯度解析失败：{e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### 逐张拍照（物理切换样本）")
cur_level = st.sidebar.radio("当前样本梯度（放好样本后拍这一张）", levels, format_func=lambda x: f"{int(x)}%")
cap_one_btn = st.sidebar.button("拍摄当前梯度（1张）")
clear_btn = st.sidebar.button("清空本轮4张")
ready = all(lv in st.session_state.captured for lv in levels)
detect_btn = st.sidebar.button("开始检测（凑齐4张后可点）", disabled=not ready)


# =========================================================
# 6) Layout
# =========================================================
st.title("智能便携式显微镜细菌识别系统")
col_video, col_info = st.columns([2, 1])

with col_video:
    st.subheader("实时影像流")
    frame_placeholder = st.empty()

with col_info:
    st.subheader("分析结果")
    result_placeholder = st.empty()
    history_expander = st.expander("本轮4张槽位", expanded=True)

# 实时预览
if st.session_state.running:
    frame_placeholder.markdown(
        f"""<div class="img-container"><img src="{video_url}" style="width:100%; border-radius:10px;" /></div>""",
        unsafe_allow_html=True,
    )
else:
    frame_placeholder.info("点击左侧“开启实时预览”查看 EB2 摄像头画面。")

# 设备状态（可选）
with col_info:
    with st.expander("设备状态（可选）", expanded=False):
        try:
            s = requests.get(status_url, timeout=1).json()
            st.json(s)
        except Exception:
            st.write("未能读取 /status（不影响视频预览）。")

# 清空
if clear_btn:
    st.session_state.captured = {}
    st.session_state.capture_session_dir = None
    st.session_state.infer_results = None
    st.session_state.last_error = ""
    st.rerun()

# 拍照
if cap_one_btn:
    if st.session_state.capture_session_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join("captures", ts)
        os.makedirs(session_dir, exist_ok=True)
        st.session_state.capture_session_dir = session_dir
        st.session_state.infer_results = None

    try:
        frame = fetch_one_frame_from_mjpeg(video_url, timeout=6.0)
        if frame is None:
            st.session_state.last_error = "抓拍失败：无法从 /video_feed 解析到有效帧。"
        else:
            fname = f"level_{int(cur_level)}.jpg"
            path = os.path.join(st.session_state.capture_session_dir, fname)
            cv2.imwrite(path, frame)
            st.session_state.captured[float(cur_level)] = path
            st.session_state.last_error = ""
            st.success(f"已拍摄：{int(cur_level)}% → {path}")
            st.rerun()
    except Exception as e:
        st.session_state.last_error = f"抓拍异常：{e}"

# 槽位显示
with history_expander:
    st.write(f"本轮目录：{st.session_state.capture_session_dir or '未开始'}")
    cols = st.columns(4)
    for i, lv in enumerate(levels):
        if lv in st.session_state.captured:
            cols[i].markdown(f"<div class='badge-ok'>✅ {int(lv)}% 已拍</div>", unsafe_allow_html=True)
            img = safe_imread(st.session_state.captured[lv])
            if img is not None:
                cols[i].image(bgr_to_rgb(img), use_container_width=True)
        else:
            cols[i].markdown(f"<div class='badge-warn'>⬜ {int(lv)}% 未拍</div>", unsafe_allow_html=True)

    if st.session_state.last_error:
        st.warning(st.session_state.last_error)

# 推理请求
if detect_btn:
    files = []
    try:
        for lv in levels:
            path = st.session_state.captured[lv]
            files.append(("images", (os.path.basename(path), open(path, "rb"), "image/jpeg")))

        data = {"levels": json.dumps([float(x) for x in levels])}
        resp = requests.post(infer_url, files=files, data=data, timeout=180)
        resp.raise_for_status()
        st.session_state.infer_results = resp.json()
        st.session_state.last_error = ""
        st.rerun()

    except Exception as e:
        st.session_state.last_error = f"推理请求失败：{e}"
        st.error(st.session_state.last_error)
    finally:
        for _, f in files:
            try:
                f[1].close()
            except Exception:
                pass


# =========================================================
# 7) Results (右侧结果栏)
# =========================================================
if st.session_state.infer_results:
    r = st.session_state.infer_results
    items = r.get("results", [])
    items_sorted = sorted(items, key=lambda x: float(x.get("level", 0)))

    # 曲线（可选：后端提供 pred_concentration 才有意义）
    df = pd.DataFrame(
        [
            {
                "level": float(it.get("level", 0)),
                "pred_conc": float(it.get("pred_concentration", 0.0)),
                "class_id": extract_class_id(it) if extract_class_id(it) is not None else np.nan,
                "class_name": class_name_from_item(it),
            }
            for it in items_sorted
        ]
    )

    line = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("level:Q", title="梯度(%)"),
        y=alt.Y("pred_conc:Q", title="预测浓度(%)"),
        tooltip=["level", "pred_conc", "class_name"],
    ).properties(height=260)

    with result_placeholder.container():
        st.success("检测完成")

        st.markdown("### 浓度曲线（如后端提供 pred_concentration）")
        st.altair_chart(line, use_container_width=True)

        # -----------------------------
        # A) 单张结果（分类ID）
        # -----------------------------
        st.markdown("### 单张结果（分类ID）")
        mcols = st.columns(4)
        for i, it in enumerate(items_sorted):
            lv = float(it.get("level", 0))
            cid = extract_class_id(it)
            cid_str = str(cid) if cid is not None else "N/A"
            mcols[i].markdown(
                f"""
                <div class="card">
                  <div class="card-title">{lv:.0f}%</div>
                  <div class="card-big">{cid_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -----------------------------
        # B) 新增：检测类别名称（你要求的“右侧结果栏再加一部分”）
        # -----------------------------
        st.markdown("### 检测类别名称（中文）")
        ncols = st.columns(4)
        for i, it in enumerate(items_sorted):
            lv = float(it.get("level", 0))
            cname = class_name_from_item(it)
            cid = extract_class_id(it)
            tag = f"{cid} - {cname}" if cid is not None else cname
            ncols[i].markdown(
                f"""
                <div class="card">
                  <div class="card-title">{lv:.0f}%</div>
                  <div class="card-name">{tag}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 右侧再放一个“图例/说明”，避免忘记映射
        st.markdown(
            """
            <div class="legend">
              <b>类别编号说明：</b><br/>
              1：对照<br/>
              2：顺铂<br/>
              3：左旋肉碱<br/>
              4：对乙酰氨基酚<br/>
              5：苯比a
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -----------------------------
        # C) 热力图
        # -----------------------------
        st.markdown("### 热力图（后端返回则用后端，否则前端生成）")
        hcols = st.columns(4)
        overlay_imgs = []

        for i, it in enumerate(items_sorted):
            lv = float(it.get("level", 0))
            cname = class_name_from_item(it)
            b64 = it.get("heatmap_png_b64", "")

            if b64:
                heat_bgr = b64_to_bgr(b64)
                if heat_bgr is None:
                    hcols[i].info(f"{lv:.0f}%：热力图解码失败")
                    continue
                hcols[i].image(
                    bgr_to_rgb(heat_bgr),
                    caption=f"{lv:.0f}% | {cname} | 后端返回",
                    use_container_width=True,
                )
                overlay_imgs.append(heat_bgr)
                continue

            if not use_fake_cam:
                hcols[i].info(f"{lv:.0f}%：后端未返回热力图")
                continue

            path = st.session_state.captured.get(lv, "")
            orig = safe_imread(path)
            if orig is None:
                hcols[i].info(f"{lv:.0f}%：找不到原图，无法生成伪热力图")
                continue

            per_seed = int(st.session_state.fake_cam_seed + int(lv) * 1000)
            _, overlay = gen_fake_gradcam_overlay(
                orig, seed=per_seed, alpha=cam_alpha, blur_ksize=cam_blur, colormap=cv2.COLORMAP_JET
            )
            if overlay is None:
                hcols[i].info(f"{lv:.0f}%：伪热力图生成失败")
                continue

            hcols[i].image(
                bgr_to_rgb(overlay),
                caption=f"{lv:.0f}% | {cname} | 前端生成",
                use_container_width=True,
            )
            overlay_imgs.append(overlay)

        # 汇总拼接
        if len(overlay_imgs) == 4:
            h = min([im.shape[0] for im in overlay_imgs])
            rs = []
            for im in overlay_imgs:
                scale = h / im.shape[0]
                w = max(1, int(im.shape[1] * scale))
                rs.append(cv2.resize(im, (w, h)))
            merged = cv2.hconcat(rs)

            st.markdown("### 汇总热力图（4 张拼接为 1 张）")
            st.image(bgr_to_rgb(merged), use_container_width=True)
        else:
            st.info("汇总热力图需要 4 张热力图均可用（后端返回或前端生成）。")
