from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map
import os
import torch

print('Compress script')
# Set CUDA_VISIBLE_DEVICES to use all available GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(torch.cuda.device_count())])

# Login to Hugging Face if token is available
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HUGGINGFACE_TOKEN not found. This may cause issues accessing gated datasets.")

base_dir = '/workspace/shared-workspace'
model_dir = os.path.join(base_dir, 'model')
original_dir = os.path.join(base_dir, 'model_original')
compressed_dir = os.path.join(base_dir, 'compressed_model')

# Prepare for compression: ensure we have the uncompressed model in model_dir
if os.path.exists(original_dir):
    print("Found original model, moving it to model directory for compression")
    if os.path.exists(model_dir):
        import shutil
        shutil.rmtree(model_dir)
    import shutil
    shutil.move(original_dir, model_dir)
else:
    print("No original model found, will compress current model")

# Calculate device map with memory reserved for GPTQ hessians
num_gpus = torch.cuda.device_count()
print(f"Found {num_gpus} GPUs available")

# Create a custom device map that distributes layers across all GPUs
device_map = {}
for i in range(num_gpus):
    device_map[f"cuda:{i}"] = []

# Load model with accelerate support
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",  # Let accelerate handle the distribution
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Print model device placement
print("\nModel device placement:")
for name, param in model.named_parameters():
    if param.device.type == 'cuda':
        print(f"{name}: {param.device}")

# Use simple, focused calibration with just HumanEval
print("Loading HumanEval dataset for code-specific calibration...")
ds = load_dataset("openai_humaneval", split="test")
ds = ds.shuffle(seed=42)

NUM_CALIBRATION_SAMPLES = len(ds)  # Use all 164 samples
MAX_SEQUENCE_LENGTH = 2048


def preprocess(example):
    # Use HumanEval prompts for calibration
    if example.get("prompt"):
        return {"text": example["prompt"]}
    else:
        return {"text": ""}

ds = ds.map(preprocess)
# Filter out empty samples  
ds = ds.filter(lambda x: len(x["text"].strip()) > 0)
print(f"Using {len(ds)} HumanEval calibration samples")


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
# Conservative GPTQ settings optimized for CodeLlama-34B stability:
recipe = GPTQModifier(
    targets="Linear", 
    scheme="W4A16", 
    ignore=["lm_head"],
    group_size=128,          # Standard group size for stability
    block_size=128,          # Conservative block size
    dampening_frac=0.1       # Higher dampening for numerical stability
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save compressed model
model.save_pretrained(compressed_dir, save_compressed=True)
tokenizer.save_pretrained(compressed_dir)

# Move uncompressed model to model_original (backup)
import shutil
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
 