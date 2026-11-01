"""Wan2.2 I2V 서버리스 핸들러 — 그림 주소 + 움직임 설명 → mp4 를 S3 에 올리고 주소를 돌려준다.

입력: image_url · prompt · negative_prompt · width · height(16의 배수) · num_frames(4n+1, 기본 81=16fps 5초)
      · num_inference_steps(기본 30) · guidance_scale(기본 3.5) · seed · fps(기본 16) · s3_folder
출력: video_url · gen_sec · width · height · num_frames
환경변수: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_S3_BUCKET / AWS_REGION · WAN_MODEL_ID(기본 Wan-AI/Wan2.2-I2V-A14B-Diffusers)
"""
import os, time, uuid, tempfile, urllib.request
from io import BytesIO
import boto3, runpod, torch
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

MODEL_ID = os.environ.get("WAN_MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
DEFAULT_NEG = "realistic, 3d render, photo, blurry, distorted face, extra fingers, deformed hands, static image, text, watermark"

# 일꾼이 켜질 때 한 번만 모델을 올린다(80GB GPU 기준 · 모자라면 CPU 로 덜어 냄)
t_load = time.time()
PIPE = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
if torch.cuda.get_device_properties(0).total_memory >= 120 * 1024**3:
    PIPE.to("cuda")
else:
    PIPE.enable_model_cpu_offload()
print(f"[wan] 모델 올림 {MODEL_ID} {round(time.time() - t_load)}초", flush=True)


def load_image(url: str) -> Image.Image:
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
    return Image.open(BytesIO(raw)).convert("RGB")


def upload_mp4(path: str, bucket: str, folder: str) -> str:
    region = os.environ.get("AWS_REGION", "ap-northeast-2")
    key = f"{folder.strip('/')}/{uuid.uuid4().hex}.mp4"
    boto3.client("s3", region_name=region).upload_file(path, bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def handler(job):
    inp = job.get("input") or {}
    try:
        if not inp.get("image_url") or not inp.get("prompt"):
            return {"error": "image_url 과 prompt 가 필요합니다."}
        bucket = inp.get("s3_bucket") or os.environ.get("AWS_S3_BUCKET")
        if not bucket:
            return {"error": "s3_bucket 또는 AWS_S3_BUCKET 환경변수가 필요합니다."}
        w, h = int(inp.get("width", 832)), int(inp.get("height", 480))
        w, h = w // 16 * 16, h // 16 * 16
        frames_n = int(inp.get("num_frames", 81))
        frames_n = (frames_n - 1) // 4 * 4 + 1
        img = load_image(inp["image_url"]).resize((w, h))
        gen = torch.Generator("cuda").manual_seed(int(inp.get("seed", 42)))
        t0 = time.time()
        frames = PIPE(image=img, prompt=inp["prompt"], negative_prompt=inp.get("negative_prompt", DEFAULT_NEG),
                      height=h, width=w, num_frames=frames_n,
                      num_inference_steps=int(inp.get("num_inference_steps", 30)),
                      guidance_scale=float(inp.get("guidance_scale", 3.5)), generator=gen).frames[0]
        gen_sec = round(time.time() - t0, 1)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "out.mp4")
            export_to_video(frames, p, fps=int(inp.get("fps", 16)))
            url = upload_mp4(p, bucket, inp.get("s3_folder", "wan-generated"))
        return {"video_url": url, "gen_sec": gen_sec, "width": w, "height": h, "num_frames": frames_n, "model": MODEL_ID}
    except Exception as e:  # 실패 내용을 그대로 돌려준다
        return {"error": f"{type(e).__name__}: {e}"}


runpod.serverless.start({"handler": handler})
