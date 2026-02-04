### P2 추가의 핵심
1. Backbone에서 stride=4 feature를 끌어올린다
2. Neck(FPN/PAN)에 P2 경로를 추가한다
3. Detect head에서 P2를 출력에 포함시킨다

---

원본 yaml 파일 (D:\SW-test\yolo+p2\.venv\Lib\site-packages\ultralytics\cfg\models\11\yolo11.yaml)을
복사하여 yolo11_p2.yaml로 저장 후 수정

- P2 헤드를 추가하면 레이어 개수가 달라져서 기존 pt를 “완벽 동일 구조”로는 못 씁니다.
- 대신 Ultralytics는 매칭되는 레이어만 부분 로드(transfer)하는 방식으로 파인튜닝이 가능합니다.
- => P2 head는 아직 학습이 안 됐고, 
pretrained weight는 P3~P5 기준으로만 최적화돼있으므로,
- 현재 테스트는 정확도 X, 경향성 확인 O

이미지 사이즈
- imgsz=640  → actual input: 384x640
- imgsz=1088 → actual input: 640x1088
- imgsz=1440 → actual input: 768x1440

가상환경 세팅 :
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics
```

