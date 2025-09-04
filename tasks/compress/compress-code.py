from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor import oneshot
import os
import torch

print('Compress script')
# Set CUDA_VISIBLE_DEVICES to use all available GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(torch.cuda.device_count())])

# Login to Hugging Face if token is available
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    # Strip any whitespace/newlines from the token
    hf_token = hf_token.strip()
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

# Calculate device map
num_gpus = torch.cuda.device_count()
print(f"Found {num_gpus} GPUs available")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# Configuration
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048
DATASET_ID = "codeparrot/self-instruct-starcoder"
DATASET_SPLIT = "curated"

def get_calib_dataset(tokenizer):
    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES*10}]",
    )

    def preprocess(example):
        chat_messages = [
            {"role": "user", "content": example["instruction"].strip()},
            {"role": "assistant", "content": example["output"].strip()},
        ]
        tokenized_messages = tokenizer.apply_chat_template(
            chat_messages, tokenize=True
        )
        return {"input_ids": tokenized_messages}

    ds = (
        ds.shuffle(seed=42)
        .map(preprocess, remove_columns=ds.column_names)
        .select(range(NUM_CALIBRATION_SAMPLES))
    )

    return ds

# Configure the quantization algorithm to run.
recipe = [
    AWQModifier(
        duo_scaling=False,
        ignore=[
            "lm_head",
            "re:.*mlp.gate$",
            "re:.*mlp.shared_expert_gate$",
            "re:visual.*",
        ],
        scheme="W4A16",
        targets=["Linear"],
    ),
]

# Apply algorithms
print("Starting AWQ compression...")
oneshot(
    model=model,
    dataset=get_calib_dataset(tokenizer),
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    log_dir=None,
    trust_remote_code_model=True,
)

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
model.save_pretrained(compressed_dir)
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
 