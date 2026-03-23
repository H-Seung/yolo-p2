import cv2
import time
from ultralytics import YOLO
from pathlib import Path

def get_unique_path(path_str):
    path = Path(path_str)

    if not path.exists():
        return str(path)

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    i = 1
    while True:
        new_path = parent / f"{stem}_{i}{suffix}"
        if not new_path.exists():
            return str(new_path)
        i += 1

# =========================
# [1] 기본 설정
# =========================
# MODEL_PATH = "../models/yolo11l_p2_bell206l_960.pt"
MODEL_PATH = "../models/BELL206L_FIXED_250717.pt"

model_info = Path(MODEL_PATH).stem
MODEL_NAME = model_info

VIDEO_PATH = "../../../media/2026-03-18 16-26-34.mkv"
OUT_PATH = get_unique_path(f"{model_info}_{Path(VIDEO_PATH).name}.mp4")
print(f"Output path: {OUT_PATH}")

CONF = 0.3       # confidence threshold (두 모델 동일하게 유지)
IMG_SIZE = 960   # 입력 해상도 (공정 비교 위해 동일 필수)

SMALL_THRESH = 0.0004   # small object 기준: bbox 면적 / frame 면적 비율

# bbox 상세 정보 표시 옵션
DRAW_BBOX_INFO = True
ONLY_SMALL_SHOW = False   # True면 small 객체만 표시

PROGRESS_STEP = 5


# =========================
# [2] 모델 로드
# =========================
model = YOLO(MODEL_PATH)


# =========================
# [3] small object 계산
# =========================
def is_small(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    return (area / (frame_w * frame_h)) < SMALL_THRESH


def count_small_objects(boxes, frame_w, frame_h):
    return sum(is_small(box, frame_w, frame_h) for box in boxes)


# =========================
# [4] 우측 상단 정보
# =========================
def draw_top_right_info(frame, model_name, fps, cum_det, cum_small):
    h, w = frame.shape[:2]

    lines = [
        model_name,
        f"SMALL_THRESH: {SMALL_THRESH:.4f}",
        f"FPS: {fps:.1f}",
        f"Cumulative Detections: {cum_det}",
        f"Cumulative Small: {cum_small}",
    ]

    y = 30
    for text in lines:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = w - tw - 20
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        y += 30

    return frame


# =========================
# [5] bbox draw (핵심 변경)
# =========================
def draw_boxes(frame, results):
    h, w = frame.shape[:2]

    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy()
    else:
        return frame, []

    for box, conf, cls in zip(boxes, confs, clss):
        x1, y1, x2, y2 = map(int, box)

        bw = x2 - x1
        bh = y2 - y1

        label = results.names[int(cls)]
        text = f"{label} {conf:.2f}"

        small_flag = is_small(box, w, h)

        # small만 표시 옵션
        if ONLY_SMALL_SHOW and not small_flag:
            continue

        # bbox
        color = (0, 0, 255) if not small_flag \
            else (0, 255, 255)   # small 이 아니면 녹색, small은 노란색
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # confidence + label
        cv2.putText(frame,
                    text,
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)

        # bbox xywh 정보 (옵션)
        if DRAW_BBOX_INFO:
            info_text = f"{bw}, {bh}, Ratio: {bw * bh / (frame_w * frame_h):.6f}"

            cv2.putText(
                frame,
                info_text,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    return frame, boxes


# =========================
# [6] main loop
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"영상 파일 열기 실패: {VIDEO_PATH}")

fps_video = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(
    OUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_video if fps_video > 0 else 30.0,
    (frame_w, frame_h)
)

scene_text = "Scene: Small object approaching"

cumulative_detections = 0
cumulative_small = 0

processed_frames = 0
last_reported_progress = -1

print(f"[START] {VIDEO_PATH}")
print(f"[INFO] Total frames: {total_frames}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()

    # inference
    results = model(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]

    frame_vis = frame.copy()
    frame_vis, boxes = draw_boxes(frame_vis, results)

    # count
    current_det = len(boxes)
    current_small = count_small_objects(boxes, frame_w, frame_h)

    cumulative_detections += current_det
    cumulative_small += current_small

    # FPS
    fps = 1.0 / (time.time() - t0)

    # scene text
    cv2.putText(frame_vis, scene_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 255), 2)

    # top right info
    frame_vis = draw_top_right_info(
        frame_vis,
        MODEL_NAME,
        fps,
        cumulative_detections,
        cumulative_small
    )

    out.write(frame_vis)

    # progress
    processed_frames += 1
    if total_frames > 0:
        progress = int((processed_frames / total_frames) * 100)

        if progress >= last_reported_progress + PROGRESS_STEP:
            print(f"[PROGRESS] {progress}% ({processed_frames}/{total_frames})")
            last_reported_progress = progress

cap.release()
out.release()

print("[DONE]")
print(f"Detections: {cumulative_detections}")
print(f"Small: {cumulative_small}")