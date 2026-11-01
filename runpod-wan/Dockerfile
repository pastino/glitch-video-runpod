# Wan2.2 I2V A14B(허깅페이스 Wan-AI/Wan2.2-I2V-A14B-Diffusers) RunPod 서버리스 — 그림 한 장 → 짧은 영상
# 베이스·diffusers 버전은 애니마 서버리스(noobai-img2img-runpod/runpod-anima)와 같게 맞춘다.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

RUN pip install --upgrade pip && pip install \
    "diffusers==0.41.0" \
    "transformers>=4.51" \
    accelerate \
    safetensors \
    sentencepiece \
    protobuf \
    ftfy \
    "imageio[ffmpeg]" \
    "huggingface_hub>=0.30" \
    hf_transfer \
    pillow \
    runpod \
    boto3 \
    requests

COPY handler.py /app/handler.py

# HuggingFace 캐시 = RunPod Cached Models 마운트 규격(애니마·WAI 서버리스와 같은 경로) · 캐시가 없으면 첫 일꾼이 받아 둔다
ENV HF_HOME=/runpod-volume/huggingface-cache
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["python", "-u", "/app/handler.py"]
