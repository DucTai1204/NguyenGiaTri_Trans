---
base_model: /root/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14/unet
library_name: peft
tags:
- lora
---

# NguyenGiaTri_Trans Model

LoRA adapter for Stable Diffusion v1-5 to generate images in Nguyen Gia Tri's Vietnamese lacquer painting style, using ControlNet for edge conditioning.

## Model Details

- **Developer:** Le Duc Tai (ID: 22050090)
- **Type:** LoRA Adapter
- **Base:** Stable Diffusion v1-5
- **Library:** PEFT
- **Repo:** https://github.com/DucTai1204/NguyenGiaTri_Trans

## Uses

Takes an input image, applies Canny edges, and generates stylized output.

### Web App
```bash
python app.py
```
Upload image at http://localhost:5000.

### Programmatic
```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from peft import PeftModel

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet)
pipe.unet = PeftModel.from_pretrained(pipe.unet, "path/to/model").merge_and_unload()

result = pipe("nguyen gia tri style...", image=canny_image, num_inference_steps=30)
```

## How to Get Started

1. Clone repo: `git clone https://github.com/DucTai1204/NguyenGiaTri_Trans`
2. Install deps: `pip install diffusers peft flask opencv-python`
3. Run app: `python app.py`
4. For training: Use `Train.ipynb` in Colab with style images.

## Training Details

- Data: Nguyen Gia Tri paintings, augmented with flips.
- Hyperparams: LR=1e-4, Steps=1500, LoRA Rank=32.
- Process: BLIP captioning, diffusion training on attention layers.

## Source Code

- `app.py`: Flask web app for image upload and generation.
- `Train.ipynb`: Colab notebook for LoRA training.
- `visualize_process.ipynb`: Notebook for testing and visualization.
- `adapter_*.safetensors`: Trained LoRA weights.

## Limitations

Trained on specific style; may not generalize. Use for creative purposes only.


