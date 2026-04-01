import os
import torch
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from peft import PeftModel # <--- QUAN TRỌNG: Dùng thư viện này để load chuẩn như Colab

app = Flask(__name__)
CORS(app)

# --- CẤU HÌNH ---
# Đường dẫn thư mục model của bạn
LORA_PATH = r"D:\Download\saved_folder\ptrienudttnt\cuoiki\nguyengiatri_model"

# Tự động chọn thiết bị (Ưu tiên GPU nếu có, không thì CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"🚀 Server đang chạy trên: {DEVICE.upper()}")

# --- LOAD MODEL (LOGIC Y HỆT COLAB) ---
print("⏳ Đang tải Model (ControlNet + Base)...")
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=DTYPE)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=DTYPE,
    safety_checker=None
).to(DEVICE)

# Tăng tốc độ
pipe.scheduler = UniPCMultistepScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

# --- KHẮC PHỤC LỖI KHÁC BIỆT: DÙNG PEFT MERGE ---
print(f"⏳ Đang trộn LoRA từ {LORA_PATH}...")
try:
    # Load LoRA vào UNet bằng PeftModel (giống Colab)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_PATH)
    # Merge thẳng vào model để đảm bảo style ăn sâu vào ảnh
    pipe.unet = pipe.unet.merge_and_unload()
    print("✅ Đã merge LoRA thành công! Chất lượng sẽ giống Colab.")
except Exception as e:
    print(f"⚠️ Lỗi load LoRA: {e}")

# Tiết kiệm RAM nếu chạy CPU
# if DEVICE == "cpu":
#     pipe.enable_model_cpu_offload()

# --- XỬ LÝ ẢNH ---
def process_ai(image_bytes):
    # 1. Đọc ảnh
    init_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # 2. Resize chia hết cho 8
    w, h = init_image.size
    # Giới hạn max size để chạy cho nhanh
    max_size = 768
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        w, h = int(w * scale), int(h * scale)
    
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    init_image = init_image.resize((new_w, new_h))
    
    # 3. Tạo Canny Map (Nét vẽ)
    image_cv = np.array(init_image)
    image_cv = cv2.Canny(image_cv, 100, 200)
    image_cv = image_cv[:, :, None]
    image_cv = np.concatenate([image_cv, image_cv, image_cv], axis=2)
    canny_image = Image.fromarray(image_cv)

    # 4. Chạy AI (Tham số y hệt Colab)
    prompt = "nguyen gia tri style, vietnamese lacquer painting, gold leaf texture, eggshell mosaic, warm tones, traditional art, masterpiece"
    negative_prompt = "photo, realistic, 3d, bad anatomy, blurry, low quality"

    result = pipe(
        prompt,
        image=canny_image,
        negative_prompt=negative_prompt,
        num_inference_steps=30,           # Giống Colab
        guidance_scale=9.0,               # Giống Colab (Độ bám style)
        controlnet_conditioning_scale=0.9 # Giống Colab (Độ giữ nét)
    ).images[0]
    
    return result

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert():
    try:
        file = request.files['image']
        if not file: return jsonify({'error': 'Chưa chọn ảnh'}), 400
        
        img_result = process_ai(file.read())
        
        buffered = BytesIO()
        img_result.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({'image_base64': img_str})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)