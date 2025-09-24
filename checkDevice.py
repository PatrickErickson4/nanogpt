import torch

def check_cuda():
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs detected: {gpu_count}")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            print(f"  GPU {i}: {name} (Compute Capability: {capability[0]}.{capability[1]})")
        current = torch.cuda.current_device()
        print(f"Current device index: {current}")
        print(f"Current device name: {torch.cuda.get_device_name(current)}")
    else:
        print("No CUDA-capable GPU detected. Running on CPU.")

if __name__ == "__main__":
    check_cuda()