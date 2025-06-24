#!/usr/bin/env python3
import argparse
import os
import subprocess
import json
import glob
import time
import requests
from pathlib import Path

def find_latest_results(output_dir):
    """Find the latest results file in the output directory."""
    # Look for results files in the output directory and its subdirectories
    results_files = glob.glob(os.path.join(output_dir, "**", "results_*.json"), recursive=True)
    if not results_files:
        return None
    # Sort by modification time and get the latest
    return max(results_files, key=os.path.getmtime)


def evaluate_model(model_dir, model_name):
    """Evaluate a single model using direct vLLM approach with specified settings"""
    task_list = ["humaneval","mbpp"]
    
    print(f"\n{'='*60}")
    print(f"🔍 EVALUATING {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Model directory: {model_dir}")
    print(f"Running tasks: {task_list}")

    # Ensure model directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory {model_dir} does not exist!")
        return None

    # Use single GPU for evaluation
    print(f"🚀 Using single GPU for evaluation")

    # Set environment variable for code evaluation
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # Set output directory
    output_dir = os.path.join(model_dir, "results")
    
    # Single GPU strategies with specified settings
    strategies = [
        ("single_gpu", f"pretrained={model_dir},tensor_parallel_size=1,dtype=float16,tokenizer=hf-internal-testing/llama-tokenizer"),
        ("single_gpu_offload", f"pretrained={model_dir},tensor_parallel_size=1,dtype=float16,tokenizer=hf-internal-testing/llama-tokenizer,cpu_offload_gb=16")
    ]

    for i, (strategy_name, model_args) in enumerate(strategies):
        print(f"\n{'='*50}")
        print(f"Strategy {i+1}/{len(strategies)}: {strategy_name}")
        print(f"{'='*50}")
        
        # Configure environment for single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Clear any NCCL environment variables (not needed for single GPU)
        for key in ["NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", 
                   "NCCL_NET_GDR_DISABLE", "NCCL_SOCKET_IFNAME", "NCCL_DEBUG", 
                   "NCCL_TIMEOUT", "NCCL_BUFFSIZE"]:
            os.environ.pop(key, None)

        # Run evaluation using direct vLLM (matching your exact command)
        cmd = [
            "lm_eval",
            "--model", "vllm",
            "--model_args", model_args,
            "--tasks", ",".join(task_list),
            "--num_fewshot", "3",  # Use 3 as specified
            "--batch_size", "auto",
            "--output_path", output_dir,
            "--confirm_run_unsafe_code"
        ]

        print("Running evaluation command:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output
        print("\nCommand output:")
        print(result.stdout)
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"✅ {strategy_name} evaluation completed successfully!")
            break  # Success - exit the strategy loop
        else:
            print(f"❌ {strategy_name} failed with return code {result.returncode}")
            if i < len(strategies) - 1:  # Not the last strategy
                print(f"🔄 Trying next strategy...")
                continue
            else:
                print("❌ All evaluation strategies failed!")
                return None

    # Find and load the latest results file
    results_file = find_latest_results(output_dir)
    try:
        if results_file:
            print(f"\n✅ Found results file: {results_file}")
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"\n📊 {model_name} Results:")
            print("-" * 30)
            model_results = {}
            for task, task_results in results['results'].items():
                print(f"\nTask: {task}")
                for metric, value in task_results.items():
                    print(f"{metric}: {value}")
                    model_results[f"{task}_{metric}"] = value
            
            return model_results
        else:
            print(f"\n❌ No results file found in {output_dir}")
            return None
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def compare_results(original_results, compressed_results):
    """Compare original and compressed model results"""
    print(f"\n{'='*60}")
    print("📈 COMPRESSION IMPACT ANALYSIS")
    print(f"{'='*60}")
    
    if not original_results or not compressed_results:
        print("❌ Cannot compare - missing results from one or both models")
        return
    
    print(f"{'Metric':<25} {'Original':<12} {'Compressed':<12} {'Change':<10} {'Impact'}")
    print("-" * 75)
    
    for metric in original_results:
        if metric in compressed_results:
            original_val = original_results[metric]
            compressed_val = compressed_results[metric]
            
            if isinstance(original_val, (int, float)) and isinstance(compressed_val, (int, float)):
                change = compressed_val - original_val
                change_pct = (change / original_val) * 100 if original_val != 0 else 0
                
                # Determine impact symbol
                if abs(change_pct) < 1:
                    impact = "✅ Minimal"
                elif change_pct > 0:
                    impact = "📈 Improved"
                elif change_pct > -5:
                    impact = "⚠️  Small drop"
                elif change_pct > -10:
                    impact = "❌ Med drop"
                else:
                    impact = "🚨 Large drop"
                
                print(f"{metric:<25} {original_val:<12.4f} {compressed_val:<12.4f} {change_pct:<10.1f}% {impact}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate compressed model performance')
    parser.add_argument('--model-dir', required=True, help='Path to the compressed model directory')
    args = parser.parse_args()

    compressed_model_dir = args.model_dir
    
    print("🚀 Starting Compressed Model Evaluation")
    print(f"Model directory: {compressed_model_dir}")
    
    # Evaluate compressed model only
    results = evaluate_model(compressed_model_dir, "COMPRESSED MODEL")
    
    if results:
        print(f"\n{'='*60}")
        print("✅ Model evaluation complete!")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("❌ Model evaluation failed!")
        print(f"{'='*60}")
        exit(1)

if __name__ == "__main__":
    main() 