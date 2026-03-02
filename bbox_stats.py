"""
지정한 데이터셋의 라벨 bbox 평균 크기 확인 스크립트
"""
import os
from pathlib import Path
import cv2
import numpy as np

LABEL_DIR = Path("/home/rs02/NAS/Dataset/Cessna_220429/labels")
IMAGE_DIR = Path("/home/rs02/NAS/Dataset/Cessna_220429/images")

widths = []
heights = []
areas = []
ratios = []

for label_file in LABEL_DIR.glob("*.txt"):
    img_file = IMAGE_DIR / (label_file.stem + ".jpg")
    if not img_file.exists():
        continue

    img = cv2.imread(str(img_file))
    h_img, w_img = img.shape[:2]

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            _, xc, yc, w, h = map(float, parts)

            w_px = w * w_img
            h_px = h * h_img
            area_px = w_px * h_px
            ratio = area_px / (w_img * h_img)

            widths.append(w_px)
            heights.append(h_px)
            areas.append(area_px)
            ratios.append(ratio)

print("===== Cessna Object Stats =====")
print(f"평균 width(px): {np.mean(widths):.2f}")
print(f"평균 height(px): {np.mean(heights):.2f}")
print(f"평균 area(px²): {np.mean(areas):.2f}")
print(f"이미지 대비 평균 비율(%): {np.mean(ratios)*100:.3f}%")


# Cessna_220429
# 평균 width(px): 44.00
# 평균 height(px): 11.80
# 평균 area(px²): 608.00
# 이미지 대비 평균 비율(%): 0.198%