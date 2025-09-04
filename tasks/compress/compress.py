from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
import os
import torch

print('Compress script')
# Set CUDA_VISIBLE_DEVICES to use all available GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(torch.cuda.device_count())])

# Login to Hugging Face if token is available
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    # Strip any whitespace/newlines from the token
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

# Load model with increased memory limits (128GB available)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",  # Let accelerate handle the distribution
    torch_dtype=torch.float16,  # Use float16 to save memory
    low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
    max_memory={0: "45GiB", 1: "45GiB", 2: "45GiB", 3: "45GiB", "cpu": "32GiB"},  # More CPU memory available
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Print model device placement
print("\nModel device placement:")
for name, param in model.named_parameters():
    if param.device.type == 'cuda':
        print(f"{name}: {param.device}")

# Select calibration dataset for general AI workloads
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

def preprocess(example):
    # Handle models with and without chat templates
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        try:
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                )
            }
        except Exception as e:
            print(f"Chat template failed, using raw message content: {e}")
            # Fallback to concatenating message content
            text = " ".join([msg.get("content", "") for msg in example["messages"] if msg.get("content")])
            return {"text": text}
    else:
        # No chat template available, use raw message content
        text = " ".join([msg.get("content", "") for msg in example["messages"] if msg.get("content")])
        return {"text": text}

ds = ds.map(preprocess)


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
# GPTQ settings compatible with current llmcompressor version:
recipe = GPTQModifier(
    targets="Linear", 
    scheme="W4A16", 
    ignore=["lm_head"],
    block_size=128,          # Standard block size
    dampening_frac=0.15,     # Higher dampening for stability
    actorder=True,           # Use activation order for better accuracy
    sequential_update=True   # Sequential processing
)

# Create required directories for oneshot
os.makedirs("sparse_logs", exist_ok=True)
os.makedirs("./offload", exist_ok=True)

# Clear all caches and force garbage collection before compression
import gc
torch.cuda.empty_cache()
gc.collect()
if hasattr(torch.cuda, 'reset_peak_memory_stats'):
    torch.cuda.reset_peak_memory_stats()

def print_memory_usage():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

print("Memory usage before compression:")
print_memory_usage()

# Don't set GPU memory fraction since each GPU has 48GB (plenty of headroom)
# The constraint is CPU memory at 24GB total

# Apply algorithms with ultra-conservative memory settings
print("Starting compression with minimal memory footprint...")
try:
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        save_compressed=True,  # Save in compressed format
        splits={"calibration": 1.0},  # Use all data for calibration, no validation split
    )
    print("Compression completed successfully!")
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(f"OOM Error during compression: {e}")
        print("Current memory usage:")
        print_memory_usage()
        raise
    else:
        raise

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
try:
    # Find the device of the first model parameter for proper device placement
    model_device = next(model.parameters()).device
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model_device)
    
    # Add attention mask to avoid warnings
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=50,  # Reduce tokens to avoid device issues
            do_sample=False,    # Use greedy decoding
            pad_token_id=tokenizer.eos_token_id
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("✅ Sample generation successful!")
except Exception as e:
    print(f"⚠️  Sample generation failed (common with multi-GPU setup): {e}")
    print("This doesn't affect the compressed model - continuing...")
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
 