# 8:2로 train/val 나눠서 경로만 txt파일로 저장
from pathlib import Path
import random

random.seed(42)
ROOT = Path("../Dataset/COCO_dataset_mjh")
CLASSES = ["airplane","bird","bus","car","cat","dog","motorcycle","person","truck"]

print(ROOT.exists())

train, val = [], []

for cls in CLASSES:
    imgs = list((ROOT/cls/"images").glob("*.*"))
    random.shuffle(imgs)
    n = int(len(imgs)*0.8)
    train += imgs[:n]
    val   += imgs[n:]

with open("./data/coco9/train.txt","w") as f:
    f.writelines(str(p)+"\n" for p in train)

with open("./data/coco9/val.txt","w") as f:
    f.writelines(str(p)+"\n" for p in val)
