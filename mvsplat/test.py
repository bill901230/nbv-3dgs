import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("GPU devices:", torch.cuda.device_count())
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
