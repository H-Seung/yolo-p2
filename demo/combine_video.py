import subprocess

left_input = "BELL206L_FIXED_250717_2026-03-18 16-26-34.mkv.mp4"
right_input = "yolo11l_p2_bell206l_960_2026-03-18 16-26-34.mkv.mp4"
output = "combine.mp4"

# 왼쪽 영상 crop
left_crop_left = 800
left_crop_right = 10

# 오른쪽 영상 crop
right_crop_left = 800
right_crop_right = 10

# 출력 높이 통일
target_height = 1080

filter_complex = (
    f"[0:v]setpts=PTS-STARTPTS,"  # [0:v] : 첫 번째 영상 / setpts=PTS-STARTPTS : 시작 타임스탬프를 0으로 맞춤
    f"crop=iw-{left_crop_left}-{left_crop_right}:ih:{left_crop_left}:0,"  # crop=iw-300:ih:150:0 : 왼쪽 150px, 오른쪽 150px 제거, 높이는 그대로, 시작점은 (150,0)
    f"scale=-1:{target_height}[v0];"  # scale=-1:720 : 높이를 720으로 맞추고, 너비는 비율에 맞게 자동 조정, [v0] : 첫 번째 영상 출력 이름
    f"[1:v]setpts=PTS-STARTPTS,"  # [1:v] : 두 번째 영상
    f"crop=iw-{right_crop_left}-{right_crop_right}:ih:{right_crop_left}:0," # crop=iw-300:ih:150:0 : 왼쪽 150px, 오른쪽 150px 제거, 높이는 그대로, 시작점은 (150,0)
    f"scale=-1:{target_height}[v1];"
    f"[v0][v1]hstack=inputs=2[v]"  # [v0]와 [v1]을 수평으로 이어붙여서 [v]라는 이름으로 출력
)

cmd = [
    "ffmpeg", "-y",
    "-i", left_input,
    "-i", right_input,
    "-filter_complex", filter_complex,
    "-map", "[v]",
    "-c:v", "h264_nvenc", # GPU 인코딩 (NVIDIA)
    # "-c:v", "libx264",  # cpu 인코딩
    "-preset", "p5",  # GPU
    # "-crf", "18",
    # "-preset", "medium",
    output
]

subprocess.run(cmd, check=True)
print(f"완료: {output}")