
import os
import requests
from tqdm import tqdm
import argparse
import sys

# Try to import config to see what's currently used
try:
    from core import config
    CONFIG_LOADED = True
except ImportError:
    CONFIG_LOADED = False

MODELS = {
    "GFPGANv1.4.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "size": 348632874,
        "type": "enhancer"
    },
    "inswapper_128.onnx": {
        "url": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        "size": 554259814,
        "type": "swapper"
    },
    "inswapper_128.fp16.onnx": {
        "url": "https://huggingface.co/netrunner-exe/Insight-Swap-models/resolve/main/inswapper_128.fp16.onnx",
        "size": 277129907, # Approximate
        "type": "swapper"
    },
}

def download_file(url, filename, expected_size=None):
    if not url:
        print(f"Skipping {filename}: No download URL available.")
        return

    local_path = os.path.join("models", filename)
    
    if os.path.exists(local_path):
        # Check size if available
        if expected_size:
            local_size = os.path.getsize(local_path)
            # Allow some tolerance for size mismatch (e.g. 1%)
            if abs(local_size - expected_size) / expected_size < 0.05:
                print(f"{filename} already exists and size matches. Skipping.")
                return
            else:
                 print(f"{filename} exists but size mismatch ({local_size} vs {expected_size}). Redownloading...")
        else:
            print(f"{filename} already exists. Skipping.")
            return

    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        with open(local_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)

def get_required_models(args):
    """Determine which models to download based on config and args."""
    
    if args.model:
         return [m for m in MODELS if m in args.model]
         
    if args.all:
        return list(MODELS.keys())

    required = [] 
    swapper_model = "inswapper_128.onnx"
    
    if CONFIG_LOADED:
        try:
            configured_swapper = config.SWAPPER_MODEL.name
            if configured_swapper in MODELS:
                swapper_model = configured_swapper
            elif "fp16" in configured_swapper and "inswapper_128.fp16.onnx" in MODELS:

                 swapper_model = "inswapper_128.fp16.onnx"
            else:
                print(f"Note: Configured swapper '{configured_swapper}' not in download list. Using default.")

        except Exception as e:
            print(f"Error reading config: {e}")

    if swapper_model not in required:
        required.append(swapper_model)
        
    return required

def main():
    parser = argparse.ArgumentParser(description="Download required models for Deep-Face-Net")
    parser.add_argument("--all", action="store_true", help="Download all available models")
    parser.add_argument("--model", nargs="+", help="Download specific model(s) by name")
    parser.add_argument("--list", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for m in MODELS:
            print(f" - {m}")
        return

    if not os.path.exists("models"):
        os.makedirs("models")
        print("Created models/ directory.")

    models_to_download = get_required_models(args)
    
    print(f"Targets: {', '.join(models_to_download)}")
    
    for filename in models_to_download:
        if filename in MODELS:
            data = MODELS[filename]
            download_file(data["url"], filename, data.get("size"))
        else:
            print(f"Warning: Model {filename} not defined in script.")
    
    print("\nDownload process completed.")

if __name__ == "__main__":
    main()
