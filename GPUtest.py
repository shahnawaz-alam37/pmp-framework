import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"\nGPU Count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        
        props = torch.cuda.get_device_properties(i)
        print(f"Total Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Max Threads Per Block: {props.max_threads_per_block}")
        
        # Current memory usage
        print(f"Allocated Memory: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"Reserved Memory: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
else:
    print("No GPU available - using CPU only")