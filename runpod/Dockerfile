# 글리치 비디오 I2V 서버리스 워커 — worker-comfyui + GGUF 노드
# 모델은 이미지에 안 굽고 네트워크 볼륨(/runpod-volume/models/...)에서 로드
FROM runpod/worker-comfyui:5.8.6-base

# AniSora V3.2 GGUF 로딩용 커스텀 노드
RUN comfy-node-install comfyui-gguf
