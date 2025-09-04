from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
import os
import shutil
from huggingface_hub import login

print('Compress script - simplified AutoGPTQ approach')

# Login to Hugging Face if token is available
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    hf_token = hf_token.strip()
    login(token=hf_token)
    print("✅ Successfully authenticated with Hugging Face")
else:
    print("Warning: HF_TOKEN not found. This may cause issues accessing gated datasets.")

base_dir = '/workspace/shared-workspace'
model_dir = os.path.join(base_dir, 'model')
original_dir = os.path.join(base_dir, 'model_original')
compressed_dir = os.path.join(base_dir, 'compressed_model')

# Prepare for compression: ensure we have the uncompressed model in model_dir
if os.path.exists(original_dir):
    print("Found original model, moving it to model directory for compression")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    shutil.move(original_dir, model_dir)
else:
    print("No original model found, will compress current model")

# Configuration
num_samples = 756
max_seq_len = 4064

tokenizer = AutoTokenizer.from_pretrained(model_dir)

def preprocess_fn(example):
    try:
        return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}
    except Exception as e:
        print(f"Chat template failed: {e}")
        # Fallback to concatenating message content
        text = " ".join([msg.get("content", "") for msg in example["messages"] if msg.get("content")])
        return {"text": text}

print(f"Loading {num_samples} samples from neuralmagic dataset...")
ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
ds = ds.shuffle().select(range(num_samples))
ds = ds.map(preprocess_fn)

print(f"Tokenizing {len(ds)} samples...")
examples = [tokenizer(example["text"], padding=False, max_length=max_seq_len, truncation=True) for example in ds]

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    model_file_base_name="model",
    damp_percent=0.1,
)

print("Loading model for quantization...")
model = AutoGPTQForCausalLM.from_pretrained(
    model_dir,
    quantize_config,
    device_map="auto",
)

print("Starting quantization...")
model.quantize(examples)
print("Quantization complete!")

print("Saving quantized model...")
model.save_pretrained(compressed_dir)
tokenizer.save_pretrained(compressed_dir)

# Move uncompressed model to model_original (backup)
shutil.move(model_dir, original_dir)

# Move compressed model to model directory (final location)
shutil.move(compressed_dir, model_dir)

print("Compression complete!")
print(f"- Compressed model saved to: {model_dir}")
print(f"- Original model backed up to: {original_dir}")

# Calculate and display size reduction
def get_directory_size(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def format_size(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"

print("\n" + "="*60)
print("📊 COMPRESSION RESULTS")
print("="*60)

original_size = get_directory_size(original_dir)
compressed_size = get_directory_size(model_dir)

print(f"Original model size:   {format_size(original_size)}")
print(f"Compressed model size: {format_size(compressed_size)}")

if original_size > 0:
    reduction_bytes = original_size - compressed_size
    reduction_percent = (reduction_bytes / original_size) * 100
    compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    
    print(f"Size reduction:        {format_size(reduction_bytes)} ({reduction_percent:.1f}%)")
    print(f"Compression ratio:     {compression_ratio:.1f}x")
else:
    print("Could not calculate size reduction")

print("="*60)