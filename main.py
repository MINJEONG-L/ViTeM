
import torch
from diffusers import StableDiffusionPipeline
import os
import time
import shutil
import json
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import inception_v3
import torchvision.transforms as TF
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import linalg
from tqdm import tqdm
from tomesd import apply_patch, remove_patch
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import math
import torch
from tomesd import merge
# 원하는 옵션 설정 (예: 줄 수, 너비 등)
torch.set_printoptions(threshold=10000, linewidth=200)
import matplotlib.pyplot as plt
import pandas as pd


from torch.profiler import profile, ProfilerActivity
parser = argparse.ArgumentParser(description='token merge image generation')
parser.add_argument('generated_image_path', type=str, help='save path')
args = parser.parse_args()

imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
crossattn = None

def get_attn2_projs(unet, down_block=0, attn_idx=0, tr_idx=0):
    target_suffix = f"down_blocks.{down_block}.attentions.{attn_idx}.transformer_blocks.{tr_idx}.attn2"
    for name, m in unet.named_modules():
        if name.endswith(target_suffix):
            return m.to_q, m.to_k, name
    raise RuntimeError(f"attn2 not found: {target_suffix}")



def load_imagenet_class_names(imagenet_root, class_index_file="imagenet_class_index.json"):
    """ImageNet 폴더에서 synset ID를 실제 클래스명으로 변환"""
    imagenet_class_index = json.load(open(class_index_file))
    class_folders = sorted(os.listdir(imagenet_root),reverse=True)

    class_names = {}
    for synset in class_folders:
        class_name = None
        for _, (synset_id, name) in imagenet_class_index.items():
            if synset_id == synset:
                class_name = name
                break
        if class_name:
            class_names[synset] = class_name
        else:
            print(f"Warning: {synset} not found in imagenet_class_index.json")

    return class_names

def load_real_imagenet_samples(imagenet_root):
    """각 클래스에서 처음 5개씩 선택하여 5000개 real 샘플 로드"""
    dataset = datasets.ImageFolder(imagenet_root, transform=imagenet_transform)
    
    class_to_images = {}
    for img_path, label in dataset.samples:
        if label not in class_to_images:
            class_to_images[label] = []
        class_to_images[label].append(img_path)

    selected_images = []
    for class_idx in sorted(class_to_images.keys()):
        images = class_to_images[class_idx][:3]
        selected_images.extend(images)

    return selected_images[:3000]  # 총 5000개 선택
def measure_flops_with_profiler(pipe, prompt, device="cuda:2"):
    """Stable Diffusion 모델에서 FLOPs 측정 (torch.profiler 사용)"""
    pipe.to(device)

    # **입력 생성 (float16으로 변환)**
    latents = torch.randn(1, 4, 64, 64, dtype=torch.float16).to(device)
    timestep = torch.randint(0, 1000, (1,), device=device).long()
    text_embeddings = pipe.text_encoder(pipe.tokenizer(prompt, return_tensors="pt").input_ids.to(device))[0]

    # **Profiler 실행 (CPU & GPU 연산 추적)**
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        record_shapes=True,
        with_flops=True  # FLOPs 계산 활성화
    ) as prof:
        with torch.no_grad():
            _ = pipe.unet(latents, timestep, text_embeddings)

    # **FLOPs 계산**
    total_flops = sum(event.self_cpu_time_total for event in prof.key_averages())

    print(f"✅ 측정된 FLOPs: {total_flops / 1e9:.2f} GFLOPs")
    return total_flops / 1e9  # GFLOPs 단위로 반환

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel

# CLIP 모델 로드
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir="/mnt/data1/hf").to("cuda:2")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir="/mnt/data1/hf")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16", cache_dir="/mnt/data1/hf")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16", cache_dir="/mnt/data1/hf").to("cuda:2")



def calculate_clip_score(image_path, prompt):
    """생성된 이미지와 프롬프트 간 CLIP Score 계산"""
    generated_image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=[prompt], images=generated_image, return_tensors="pt", padding=True).to("cuda:2")

    with torch.no_grad():
        image_features = clip_model.get_image_features(inputs["pixel_values"])
        text_features = clip_model.get_text_features(inputs["input_ids"])

        # 벡터 정규화
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine Similarity 계산
        clip_score = (image_features @ text_features.T).squeeze().item()

    return clip_score

def generate_images(pipe, class_names, save_dir, generator, config=None):
    """프롬프트 순서대로 이미지 생성"""
    global block_counter
    global crossattn
    total_flops = 0
    sampled_count = 0  # FLOPs 측정된 샘플 개수
    image_count = 0
    flops_limit = 50
    maxclip = 0
    print("Starting generation...")  #
    # start_time = time.time()
    
    
    os.makedirs(save_dir, exist_ok=True)
    # total_vram_usage = 0  # 한 장당 VRAM 사용량 추적
    total_images = 0  # 생성된 총 이미지 수
    clip_scores = []
    kkk = 0
  
    for synset, class_name in tqdm(class_names.items(), desc="Generating Images"):
        if not hasattr(pipe, "_tome_info"):
            pipe._tome_info = {}

        pipe._tome_info["block_counter"] = 0 
        prompt = f"A high quality photograph of {class_name}."

        if not hasattr(pipe, "_tome_info"):
            pipe._tome_info = {}
        with torch.no_grad():
            text_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda:2")
            final_layer_norm = pipe.text_encoder.text_model.final_layer_norm  # type: torch.nn.LayerNorm
            text_embeddings = pipe.text_encoder(**text_inputs).last_hidden_state  # [1, D]
            pipe._tome_info["text_embedding"] = text_embeddings
            pipe._tome_info["final_layer_norm"] = final_layer_norm

        if config is not None:
            apply_patch(
            pipe,
            ratio=config["ratio"],
            max_downsample=config["max_downsample"],
            merge_attn=config["merge_attn"],
            merge_crossattn=config["merge_crossattn"],
            merge_mlp=config["merge_mlp"],
            text_embedding = text_embeddings,
            final_layer_norm = final_layer_norm,
            text_weight = config["text_weight"],
            q_proj = config["q_proj"],
            k_proj = config["k_proj"]
            )
            print("Patch applied successfully")  # 패치 적용 완료
        else:
            remove_patch(pipe)
        
        pipe.scheduler.set_timesteps(num_inference_steps=50)
        if pipe.scheduler.timesteps is not None:
            initial_timestep = pipe.scheduler.timesteps[0]
        else:
            raise ValueError("Scheduler timesteps is still None after set_timesteps!")
        save_path = os.path.join(save_dir, f"{synset}_1.png")
            

        if os.path.exists(save_path):
            print(f"Skipping {save_path} - file already exists")
            continue

        torch.cuda.synchronize()

        start_time = time.time()
        images = pipe(
            prompt,
            generator=generator,
            height=512,
            width=512,
            num_inference_steps=50,
            num_images_per_prompt=3,
            guidance_scale = 7.5
        ).images
        end_time = time.time()

        torch.cuda.synchronize()

        after_allocated = torch.cuda.memory_allocated() / 1e9
        total_images += len(images)
        for i, image in enumerate(images):
            save_path = os.path.join(save_dir, f"{synset}_{i}.png")
            image.save(save_path)

            clip_score = calculate_clip_score(save_path, prompt)
            clip_scores.append(clip_score)
            if maxclip <= clip_score:
                maxclip = clip_score
            print("clip score:  ", clip_score)
        flush_rep_logs("rep_logs/rep_positions.jsonl")
        print(f"한 장의 이미지 생성 시간: {end_time - start_time:.2f} 초")
    avg_clip_score = sum(clip_scores) / len(clip_scores) if clip_scores else 0
    print(f"평균 CLIP Score: {avg_clip_score:.4f}")
    return maxclip


def generate_images_from_captions(pipe, prompts, save_dir, generator, config=None):
    os.makedirs(save_dir, exist_ok=True)
    global crossattn
    print("Generating images from COCO captions...")

    all_s_im = []   # 프롬프트당 s/im
    all_clip = []   # 전체 이미지의 CLIP 점수

    max_clip_score = 0

    for _, item in enumerate(tqdm(prompts, desc="Generating")):
        image_id = item["image_id"]
        caption = item["caption"]
        crossattn = None

        file_name = f"{image_id}"
        save_path = os.path.join(save_dir)
        first_target = os.path.join(save_path, f"{file_name}_0.png")

        # 이미 앞에서 생성한 적 있으면 스킵
        if os.path.exists(first_target):
            print(f"Skipping {file_name} (already exists)")
            continue

        prompt = caption

        if not hasattr(pipe, "_tome_info"):
            pipe._tome_info = {}

        with torch.no_grad():
            text_inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda:2")
            text_embeddings = pipe.text_encoder(**text_inputs).last_hidden_state
            pipe._tome_info["text_embedding"] = text_embeddings

        num_images_per_prompt = 3

        images_generated = 0
        total_time_for_prompt = 0.0

        for i in range(num_images_per_prompt):
            start_time = time.time()
            images = pipe(
                prompt,
                generator=generator,
                height=512,
                width=512,
                num_inference_steps=50,
                num_images_per_prompt=1,
                guidance_scale=7.5,
            ).images
            end_time = time.time()

            # 저장 + CLIP
            out_path = os.path.join(save_path, f"{file_name}_{i}.png")
            images[0].save(out_path)

            clip_score = calculate_clip_score(out_path, prompt)
            all_clip.append(clip_score)
            if max_clip_score <= clip_score:
                max_clip_score = clip_score

            # 시간/개수 누적
            elapsed = end_time - start_time
            total_time_for_prompt += elapsed
            images_generated += 1

            print(f"clip score: {clip_score:.4f}")
            print(f"한 장의 이미지 생성 시간: {elapsed:.2f} 초")

        # --- 프롬프트당 s/im 기록 ---
        if images_generated > 0:
            s_im = total_time_for_prompt / images_generated
            all_s_im.append(s_im)

    # ---- 요약 리포트 ----
    s_im_arr = np.array(all_s_im, dtype=np.float64) if len(all_s_im) else np.array([float('nan')])
    clip_arr = np.array(all_clip, dtype=np.float64) if len(all_clip) else np.array([float('nan')])

    report = {
        "prompts": len(all_s_im),
        "s/im_mean_over_prompts": float(np.nanmean(s_im_arr)),
        "s/im_std_over_prompts": float(np.nanstd(s_im_arr)),
        "clip_mean": float(np.nanmean(clip_arr)),
        "clip_max": float(np.nanmax(clip_arr)),
    }

    print("\n===== SUMMARY =====")
    print(f"prompts: {report['prompts']}")
    print(f"s/im over prompts: {report['s/im_mean_over_prompts']:.4f} ± {report['s/im_std_over_prompts']:.4f}")
    print(f"CLIP mean: {report['clip_mean']:.4f} | max: {report['clip_max']:.4f}")

    return report


from pathlib import Path
def calculate_flops(pipe, image_tensor, steps=25, resolution=512, device="cuda:2"):
    """Stable Diffusion 모델의 FLOPs 계산"""
    print("Calculating FLOPs using ImageNet data...")

    # 이미지를 latent space로 변환
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        latents = pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor  # Scaling 적용

        # FLOPs 측정
        flops_analysis = FlopCountAnalysis(
            pipe.unet,
            (latents, torch.tensor([steps]).to(device), torch.randn(1, 77, 768).to(device))
        )

    flops = flops_analysis.total()
    print(f"Total FLOPs: {flops / 1e9:.2f} GFLOPs")
    return flops / 1e9  # GFLOPs 단위로 반환


class OrderedImageDataset(Dataset):
    """이미지를 지정된 순서대로 로드하는 데이터셋"""
    def __init__(self, image_dir, image_files):
        self.image_paths = [os.path.join(image_dir, x) for x in image_files]
        
        self.transforms = TF.Compose([
            TF.Resize(299),
            TF.CenterCrop(299),
            TF.ToTensor(),
            TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            return self.transforms(image)
        except Exception as e:
            print(f"Error loading image: {self.image_paths[idx]}")
            raise e

def copy_ordered_images(pairs, src_dir, dst_dir):
    """원본 이미지를 순서를 유지하며 복사"""
    os.makedirs(dst_dir, exist_ok=True)
    copied_files = []
    
    for image_file, _ in pairs:
        src_path = os.path.join(src_dir, image_file)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(dst_dir, image_file))
            copied_files.append(image_file)
    
    return copied_files


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)

# 1. 전역 변수 정의
captured_keys = {}

# 2. hook 함수 정의
def capture_to_k(module, input, output):
    captured_keys[module] = output.detach()  # [1, 77, 320]

# 3. hook 등록 함수
def register_cross_attention_hooks(unet):
    handles = []
    
    for name, module in unet.named_modules():
        if hasattr(module, "attn2") and hasattr(module.attn2, "to_k"):
            handle = module.attn2.to_k.register_forward_hook(capture_to_k)
            handles.append(handle)

    return handles


def smooth_logs(cs_logs, window=20):
    def _smooth(arr):
        arr = np.array(arr)
        # 구간 평균: window 크기 단위로 잘라서 평균
        n = len(arr) // window
        arr = arr[:n*window].reshape(n, window)
        return arr.mean(axis=1)
    return {
        "pre": _smooth(cs_logs["pre"]),
        "post": _smooth(cs_logs["post"]),
        "delta": _smooth(cs_logs["delta"]),
    }

def plot_cs_logs(cs_logs, title="Cross-Similarity Before vs After Merge", save_path=None, window=20):
    logs_smoothed = smooth_logs(cs_logs, window=window)
    steps = np.arange(len(logs_smoothed["pre"])) * window + 100

    plt.figure(figsize=(8, 5))
    plt.plot(steps, logs_smoothed["pre"], label="Pre-merge avg", linestyle="--")
    plt.plot(steps, logs_smoothed["post"], label="Post-merge avg", linestyle="-")
    plt.plot(steps, logs_smoothed["delta"], label="Δ (Post-Pre)", linestyle=":")
    plt.xlabel(f"Merge step (x{window})")
    plt.ylim(0.0, 0.25)                  # y축 범위를 0.0 ~ 0.25로 고정
    plt.yticks(np.arange(0.0, 0.26, 0.05))  # 0.05 단위로 눈금 표시
    plt.ylabel("Cross-Similarity")
    plt.legend(loc="center right")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def calculate_fid(real_images, fake_images, device='cuda:2'):
    """FID 계산"""
    print("Calculating FID...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = inception_v3(pretrained=True)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    
    real_loader = DataLoader(ImageDataset(real_images, imagenet_transform), batch_size=32, shuffle=False, num_workers=4)
    fake_loader = DataLoader(ImageDataset(fake_images, imagenet_transform), batch_size=32, shuffle=False, num_workers=4)
    
    features_real, features_fake = [], []
    
    with torch.no_grad():
        for batch in tqdm(real_loader, desc="Processing Real Images"):
            features_real.append(model(batch.to(device)).cpu().numpy())
        for batch in tqdm(fake_loader, desc="Processing Fake Images"):
            features_fake.append(model(batch.to(device)).cpu().numpy())
    
    features_real = np.concatenate(features_real, axis=0)
    features_fake = np.concatenate(features_fake, axis=0)
    
    mu_real, sigma_real = features_real.mean(axis=0), np.cov(features_real, rowvar=False)
    mu_fake, sigma_fake = features_fake.mean(axis=0), np.cov(features_fake, rowvar=False)
    
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
    return float(fid)


if __name__ == "__main__":
    generator = torch.Generator("cuda:2").manual_seed(42)
    model_id = "CompVis/stable-diffusion-v1-4"
    steps = 25
    resolution = 512

    
    # 모델 설정
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        safety_checker=None
    ).to("cuda:2")
    q_proj, k_proj, attn_name = get_attn2_projs(pipe.unet, down_block=0, attn_idx=1, tr_idx=0)
    print("[use attn2]", attn_name)

    window_configs = {
        "name": "adaptive_anchor",
        "use_rand": False,
        "ratio":0.5,
        "max_downsample": 1,
        "merge_attn": True,
        "merge_crossattn": True,
        "merge_mlp": False,
    }

    pipe.set_progress_bar_config(disable=True)

    imagenet_root = "./imagenet-val"
    real_images_path = load_real_imagenet_samples(imagenet_root)
    fake_images_path = f"tome_images_{args.generated_image_path}"
    class_names = load_imagenet_class_names(imagenet_root)
    
    kk= generate_images(pipe, class_names, fake_images_path, generator, window_configs)
    
    print(kk)
    from tomesd.merge import FLOP_LOG
    total_flops = FLOP_LOG["sim_local"] + FLOP_LOG["sim_text"]
    gflops = total_flops / 1e9
    hits = FLOP_LOG.get("cache_hit", 0)
    miss = FLOP_LOG.get("cache_miss", 0)
    hit_rate = (hits / (hits + miss)) * 100 if (hits + miss) > 0 else 0.0

    print(f"[FLOPs] similarity (local+text): {gflops:.2f} GFLOPs"
        f" | local={FLOP_LOG['sim_local']/1e9:.2f}G"
        f" | text={FLOP_LOG['sim_text']/1e9:.2f}G")
    print(f"[CACHE] hit={hits}, miss={miss}, hit_rate={hit_rate:.1f}%")

##VITERM
