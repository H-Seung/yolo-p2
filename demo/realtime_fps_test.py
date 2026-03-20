import cv2
import time
import threading
import logging
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# =========================
# 기본 설정
# =========================
# MODEL_PATH = "../models/yolo11l_p2_bell206l_960.pt"
MODEL_PATH = "../models/BELL206L_FIXED_250717.pt"
VIDEO_PATH = "../../../media/2026-03-18 16-26-34.mkv"

model_info = Path(MODEL_PATH).stem
log_path = Path(f"{model_info}_FPS+test.log")

CONF = 0.3
IMG_SIZE = 960

# =========================
# 로그 설정 (append 모드)
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="a"),   # ✅ append
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()

# =========================
# 실행 구분 헤더 추가
# =========================
run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

logger.info("")  # 한 줄 띄움
logger.info("=" * 60)
logger.info(f"[NEW RUN] {run_time}")
logger.info("=" * 60)

# =========================
# 공유 변수
# =========================
latest_frame = None
latest_frame_id = -1
frame_lock = threading.Lock()

running = True
capture_finished = False

# =========================
# 모델 로드
# =========================
model = YOLO(MODEL_PATH)

# =========================
# Capture Thread
# =========================
def capture_thread(cap):
    global latest_frame, latest_frame_id, running, capture_finished

    frame_id = 0
    while running:
        ret, frame = cap.read()
        if not ret:
            capture_finished = True
            break

        with frame_lock:
            latest_frame = frame
            latest_frame_id = frame_id

        frame_id += 1

    capture_finished = True


# =========================
# main
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"영상 파일 열기 실패: {VIDEO_PATH}")

t_cap = threading.Thread(target=capture_thread, args=(cap,), daemon=True)
t_cap.start()

last_inferred_id = -1
infer_count = 0
start_time = time.time()
last_log_time = start_time

logger.info("[START] Real-time inference FPS test")
logger.info(f"[INFO] Model: {model_info}")
logger.info(f"[INFO] Video: {VIDEO_PATH}")

while True:
    with frame_lock:
        if latest_frame is None:
            frame = None
            frame_id = -1
        else:
            frame = latest_frame.copy()
            frame_id = latest_frame_id

    if frame is None:
        if capture_finished:
            break
        time.sleep(0.001)
        continue

    if frame_id == last_inferred_id:
        if capture_finished:
            time.sleep(0.001)
        else:
            time.sleep(0.0005)
        continue

    # =========================
    # 추론
    # =========================
    _ = model(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]

    infer_count += 1
    last_inferred_id = frame_id

    now = time.time()
    elapsed_total = now - start_time
    elapsed_log = now - last_log_time

    # 1초마다 로그
    if elapsed_log >= 1.0:
        fps = infer_count / elapsed_total if elapsed_total > 0 else 0.0
        logger.info(
            f"[INFO] inferred_frames={infer_count}, avg_infer_fps={fps:.2f}"
        )
        last_log_time = now

    if capture_finished and frame_id >= latest_frame_id:
        break

running = False
t_cap.join(timeout=1.0)
cap.release()

total_time = time.time() - start_time
final_fps = infer_count / total_time if total_time > 0 else 0.0

logger.info("[DONE]")
logger.info(f"[RESULT] Total inferred frames: {infer_count}")
logger.info(f"[RESULT] Average inference FPS: {final_fps:.2f}")
logger.info("=" * 60)
logger.info("")  # 실행 끝나고 한 줄 띄움