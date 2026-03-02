# 사전학습모델(custom 학습x)로 640, 1088, 1440 사이즈 이미지에 추론하고 elapsed time, vram 측정하여 csv로 저장
from ultralytics import YOLO
from pathlib import Path
import random
import time
import csv
import torch

# =========================
# 설정
# =========================
MODEL_YAML = "models/yolo11x_p2.yaml"
MODEL_YAML = "yolo11x.yaml"
PRETRAIN = "yolo11x.pt"

IMAGE_DIR = Path("Z:/home/rs02/NAS/Dataset/Cessna_220429_fhd/images")   # 테스트 이미지 폴더
OUTPUT_DIR = Path("runs_p2_test")
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZES = [640, 1088, 1440]   # 단일 imgsz만 사용
NUM_SAMPLES = 30          # 이미지 샘플 개수
CONF = 0.25
DEVICE = 0

CSV_PATH = OUTPUT_DIR / "benchmark_yolo11_p2.csv"

# =========================
# 이미지 샘플링
# =========================
random.seed(42)
all_images = sorted(list(IMAGE_DIR.glob("*.*")))
assert len(all_images) > 0, "이미지 폴더에 파일이 없습니다."

if len(all_images) > NUM_SAMPLES:
    sample_images = random.sample(all_images, NUM_SAMPLES)
else:
    sample_images = all_images

print(f"[INFO] Using {len(sample_images)} / {len(all_images)} images for benchmark")

# =========================
# 모델 로드
# =========================
model = YOLO(MODEL_YAML).load(PRETRAIN)
model.info()

# =========================
# CSV 헤더 (파일 존재할 때는 생략)
# =========================
write_header = not CSV_PATH.exists()

with open(CSV_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "model_yaml",
            "imgsz",
            "elapsed_sec",
            "peak_vram_mb",
            "num_images",
            "gpu_name",
        ])

# =========================
# 이미지 사이즈별 벤치마크
# =========================
for imgsz in IMG_SIZES:
    print(f"\n===== Benchmark imgsz={imgsz} =====")

    # =========================
    # Warm-up (측정 제외)
    # =========================
    _ = model.predict(
        source=[str(sample_images[0])],  # 1장만
        imgsz=imgsz,
        conf=CONF,
        rect=True,
        device=DEVICE,
        save=False,
        verbose=False
    )
    torch.cuda.synchronize()

    # =========================
    # 실제 측정 시작
    # =========================

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t0 = time.time()

    results = model.predict(
        source=[str(p) for p in sample_images],
        imgsz=imgsz,
        conf=CONF,
        rect=True,
        device=DEVICE,
        save=False,
        stream=False,
        verbose=False
    )

    torch.cuda.synchronize()
    t1 = time.time()
    elapsed = t1 - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"[RESULT] model_yaml={Path(MODEL_YAML).name}")
    print(f"         imgsz={imgsz}")
    print(f"         elapsed time: {elapsed:.3f} sec")
    print(f"         peak VRAM   : {peak_vram:.1f} MB")

    # =========================
    # CSV 저장
    # =========================
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            Path(MODEL_YAML).name,
            imgsz,
            f"{elapsed:.4f}",
            f"{peak_vram:.1f}",
            len(sample_images),
            torch.cuda.get_device_name(DEVICE),
        ])

print(f"\n[INFO] Benchmark results saved to: {CSV_PATH}")

