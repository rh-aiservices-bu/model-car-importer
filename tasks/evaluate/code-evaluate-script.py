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

def try_start_vllm_server(model_dir, port, strategy, num_gpus):
    """Try to start vLLM server with a specific strategy"""
    
    if strategy == "tensor_parallel":
        print(f"🚀 Attempt 1: Tensor parallelism with {num_gpus} GPUs (NCCL)")
        # Configure NCCL environment
        nccl_env = {
            "NCCL_IB_DISABLE": "1",
            "NCCL_P2P_DISABLE": "1", 
            "NCCL_SHM_DISABLE": "1",
            "NCCL_NET_GDR_DISABLE": "1",
            "NCCL_SOCKET_IFNAME": "lo",
            "NCCL_DEBUG": "INFO",
            "NCCL_TIMEOUT": "3600",
            "NCCL_BUFFSIZE": "1048576",
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(num_gpus))
        }
        os.environ.update(nccl_env)
        
        server_cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_dir,
            "--port", str(port),
            "--host", "0.0.0.0",
            "--tensor-parallel-size", str(num_gpus)
        ]
        
    elif strategy == "pipeline_parallel":
        print(f"🔄 Attempt 2: Pipeline parallelism with {min(num_gpus, 4)} stages (no NCCL)")
        # Clear NCCL environment
        for key in ["NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", 
                   "NCCL_NET_GDR_DISABLE", "NCCL_SOCKET_IFNAME", "NCCL_DEBUG", 
                   "NCCL_TIMEOUT", "NCCL_BUFFSIZE"]:
            os.environ.pop(key, None)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
        
        server_cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_dir,
            "--port", str(port),
            "--host", "0.0.0.0",
            "--pipeline-parallel-size", str(min(num_gpus, 4))
        ]
        
    elif strategy == "single_gpu_offload":
        print(f"🔄 Attempt 3: Single GPU with CPU offloading")
        # Use only first GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        server_cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_dir,
            "--port", str(port),
            "--host", "0.0.0.0",
            "--cpu-offload-gb", "16"
        ]
        
    elif strategy == "single_gpu":
        print(f"🔄 Attempt 4: Single GPU only")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        server_cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_dir,
            "--port", str(port),
            "--host", "0.0.0.0"
        ]
        
    else:
        return None
    
    print("Starting vLLM server with command:", " ".join(server_cmd))
    return subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def wait_for_server_ready(server_process, port, strategy, timeout_minutes=20):
    """Wait for server to be ready with health checking"""
    print(f"⏳ Waiting for {strategy} server to start (timeout: {timeout_minutes}m)...")
    
    start_time = time.time()
    model_loading_started = False
    last_output_check = 0
    
    for attempt in range(timeout_minutes * 12):  # Check every 5 seconds
        # Check if process has crashed
        if server_process.poll() is not None:
            print("❌ vLLM server process crashed!")
            print("Server output:")
            try:
                stdout, _ = server_process.communicate(timeout=5)
                print(stdout)
            except:
                print("Could not retrieve server output")
            return False
            
        # Try to connect to server
        try:
            response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
            if response.status_code == 200:
                elapsed = time.time() - start_time
                print(f"✅ vLLM server is ready! (took {elapsed/60:.1f}m)")
                return True
        except Exception as e:
            if attempt == 0:
                print(f"Connection attempt failed: {e}")
        
        # Show progress every 60 seconds
        if attempt % 12 == 0 and attempt > 0:
            elapsed_minutes = (attempt * 5) // 60
            print(f"⏳ Still waiting... ({elapsed_minutes}m elapsed)")
            
            # Show server output periodically
            if attempt >= 12 and attempt - last_output_check >= 12:  # Every minute after first minute
                last_output_check = attempt
                try:
                    import fcntl
                    fd = server_process.stdout.fileno()
                    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                    
                    try:
                        output = server_process.stdout.read(2000)
                        if output:
                            output_str = output.strip()
                            # Check for progress indicators
                            if "Starting to load model" in output_str and not model_loading_started:
                                model_loading_started = True
                                print("🔄 Model loading has started...")
                            elif "Loading weights took" in output_str:
                                print("⚡ Model weights loaded!")
                            elif "Model loading took" in output_str:
                                print("✅ Model loading completed!")
                            elif "NCCL WARN" in output_str or "NCCL INFO" in output_str:
                                print("📡 NCCL communication active...")
                            elif "ERROR" in output_str or "Worker proc" in output_str and "died" in output_str:
                                print("⚠️  Errors detected in server output")
                    except BlockingIOError:
                        pass  # No output available
                        
                except Exception as e:
                    pass  # Ignore output reading errors
        
        time.sleep(5)
    
    print(f"❌ Server failed to start within {timeout_minutes} minutes")
    return False

def start_vllm_server(model_dir, num_gpus, port=8000):
    """Start vLLM server with robust multi-stage fallback"""
    
    strategies = []
    
    if num_gpus > 1:
        strategies.extend([
            "tensor_parallel",      # Best performance, needs NCCL
            "pipeline_parallel",    # Good performance, no NCCL  
            "single_gpu_offload",   # CPU offloading for large models
            "single_gpu"           # Last resort
        ])
    else:
        strategies.extend([
            "single_gpu_offload",   # Try offloading first for large models
            "single_gpu"           # Standard single GPU
        ])
    
    for i, strategy in enumerate(strategies):
        print(f"\n{'='*50}")
        print(f"Strategy {i+1}/{len(strategies)}: {strategy}")
        print(f"{'='*50}")
        
        # Start server with current strategy
        server_process = try_start_vllm_server(model_dir, port, strategy, num_gpus)
        if not server_process:
            continue
            
        # Wait for server to be ready
        timeout = 30 if strategy == "tensor_parallel" else 20  # More time for tensor parallel
        if wait_for_server_ready(server_process, port, strategy, timeout):
            print(f"🎉 Successfully started server with {strategy}")
            return server_process
        else:
            print(f"❌ {strategy} failed, trying next strategy...")
            try:
                server_process.terminate()
                server_process.wait(timeout=10)
            except:
                server_process.kill()
    
    print("❌ All strategies failed!")
    return None

def evaluate_model(model_dir, model_name):
    """Evaluate a single model using vLLM server approach"""
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

    # Detect GPU count
    try:
        import torch
        num_gpus = torch.cuda.device_count()
        print(f"🚀 Detected {num_gpus} GPUs for evaluation")
    except ImportError:
        print("⚠️  Could not detect GPU count, using single GPU")
        num_gpus = 1

    # Set output directory
    output_dir = os.path.join(model_dir, "evaluation_results")
    
    # Choose a unique port for this evaluation
    port = 8000
    
    # Start vLLM server
    server_process = start_vllm_server(model_dir, num_gpus, port)
    if not server_process:
        return None
    
    try:
        # Run evaluation against vLLM server
        cmd = [
            "lm_eval",
            "--model", "openai-completions",
            "--model_args", f"base_url=http://localhost:{port}/v1,api_key=EMPTY,model={model_name}",
            "--tasks", ",".join(task_list),
            "--num_fewshot", "3",
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

        if result.returncode != 0:
            print(f"\n❌ Evaluation failed with return code {result.returncode}")
            return None
            
    finally:
        # Always cleanup the server
        print("🛑 Stopping vLLM server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("⚠️  Force killing vLLM server...")
            server_process.kill()

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
    parser = argparse.ArgumentParser(description='Compare original and compressed model performance')
    parser.add_argument('--model-dir', required=True, help='Path to the compressed model directory')
    args = parser.parse_args()

    # Define model directories
    compressed_model_dir = args.model_dir
    original_model_dir = os.path.join(os.path.dirname(compressed_model_dir), "model_original")
    
    print("🚀 Starting Model Comparison Evaluation")
    print(f"Compressed model: {compressed_model_dir}")
    print(f"Original model: {original_model_dir}")
    
    # Evaluate original model first
    original_results = evaluate_model(original_model_dir, "ORIGINAL MODEL")
    
    # Evaluate compressed model
    compressed_results = evaluate_model(compressed_model_dir, "COMPRESSED MODEL")
    
    # Compare results
    compare_results(original_results, compressed_results)
    
    print(f"\n{'='*60}")
    print("✅ Model comparison evaluation complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 