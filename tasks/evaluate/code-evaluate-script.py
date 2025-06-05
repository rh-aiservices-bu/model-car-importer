#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import json
import glob
from pathlib import Path

def find_latest_results(output_dir):
    """Find the latest results file in the output directory."""
    # Look for results files in the output directory and its subdirectories
    results_files = glob.glob(os.path.join(output_dir, "**", "results_*.json"), recursive=True)
    if not results_files:
        return None
    # Sort by modification time and get the latest
    return max(results_files, key=os.path.getmtime)

def main():
    parser = argparse.ArgumentParser(description='Evaluate a model using lm-evaluation-harness')
    parser.add_argument('--model-dir', required=True, help='Path to the model directory')
    args = parser.parse_args()

    # Convert comma-separated tasks to list
    task_list = "humaneval,mbpp"

    print(f"Evaluating model at {args.model_dir}")
    print(f"Running tasks: {task_list}")

    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)

    # Set output directory
    output_dir = os.path.join(args.model_dir, "evaluation_results")

    # Run evaluation using subprocess to avoid import issues
    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", f"pretrained={args.model_dir},add_bos_token=True",
        "--tasks", ",".join(task_list),
        "--num_fewshot", "3",
        "--batch_size", "auto",
        "--output_path", output_dir,
        "--confirm_run_unsafe_code"
    ]

    print("Running command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print("\nCommand output:")
    print(result.stdout)
    if result.stderr:
        print("\nErrors:")
        print(result.stderr)

    # Check if evaluation was successful
    if result.returncode != 0:
        print(f"\nEvaluation failed with return code {result.returncode}")
        sys.exit(result.returncode)

    # Find and load the latest results file
    results_file = find_latest_results(output_dir)
    try:
        if results_file:
            print(f"\nFound results file: {results_file}")
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print("\nEvaluation Results:")
            print("------------------")
            for task, task_results in results['results'].items():
                print(f"\nTask: {task}")
                for metric, value in task_results.items():
                    print(f"{metric}: {value}")
        else:
            print(f"\nNo results file found in {output_dir}")
            print("This might indicate that the evaluation failed to complete successfully.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 