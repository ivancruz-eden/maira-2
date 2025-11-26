"""
Check GPU availability and memory for MAIRA-2 testing.
"""
import torch


def main():
    print("=" * 80)
    print("GPU STATUS CHECK")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("\n❌ No CUDA-capable GPU detected")
        print("The model will run on CPU (very slow)")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"\n✓ Detected {num_gpus} GPU(s)\n")
    
    total_memory = 0
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        gpu_name = props.name
        gpu_memory_gb = props.total_memory / 1024**3
        total_memory += gpu_memory_gb
        
        # Get current memory usage
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        free = gpu_memory_gb - allocated
        
        print(f"GPU {i}: {gpu_name}")
        print(f"  Total Memory: {gpu_memory_gb:.2f} GB")
        print(f"  Allocated:    {allocated:.2f} GB")
        print(f"  Reserved:     {reserved:.2f} GB")
        print(f"  Free:         {free:.2f} GB")
        print()
    
    print("-" * 80)
    print(f"Total GPU Memory: {total_memory:.2f} GB")
    print("-" * 80)
    
    # Estimate model requirements
    model_size_gb = 14  # ~7B params * 2 bytes (FP16) ≈ 14GB
    print(f"\nMAIRA-2 Model Requirements (FP16): ~{model_size_gb} GB")
    print(f"Available across all GPUs: {total_memory:.2f} GB")
    
    if total_memory >= model_size_gb:
        print(f"\n✓ Sufficient memory for multi-GPU distribution")
        per_gpu = model_size_gb / num_gpus
        print(f"  Expected usage per GPU: ~{per_gpu:.2f} GB")
    else:
        print(f"\n⚠ Warning: Total GPU memory may be insufficient")
        print(f"  Consider using CPU or reducing precision")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
