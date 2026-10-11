import torch
from diffusers import TextToVideoSDPipeline
from diffusers.utils import export_to_video

# Ini AI open source, nanti jadi punya bos
print("Loading AI Tuyul Bekas...")

pipe = TextToVideoSDPipeline.from_pretrained(
    "damo-vilab/modelscope-damo-text-to-video-synthesis",
    torch_dtype=torch.float16
)
pipe.to("cuda")

def buat_video_tuyul(prompt):
    print(f"Bikin video: {prompt}")
    video_frames = pipe(prompt, num_frames=24).frames
    export_to_video(video_frames, "tuyul_bekas.mp4")
    return "tuyul_bekas.mp4 berhasil dibikin bos!"

# Contoh pakai
buat_video_tuyul("cute green tuyul doll in bamboo basket waving, funny, market background")
