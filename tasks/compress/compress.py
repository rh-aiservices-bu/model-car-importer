from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map
import os
import torch

# Set CUDA_VISIBLE_DEVICES to use all available GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(torch.cuda.device_count())])

base_dir = '/workspace/shared-workspace'
model_dir = os.path.join(base_dir, 'model')
compressed_dir = os.path.join(base_dir, 'compressed_model')
original_dir = os.path.join(base_dir, 'model_original')

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

# Select calibration dataset.
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
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


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
#   * quantize the weights to 4 bit with GPTQ with a group size 128
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

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

# Save to disk compressed.
SAVE_DIR = compressed_dir
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
os.rename(model_dir, original_dir)

# Then rename the compressed directory to model
os.rename(compressed_dir, model_dir)
 