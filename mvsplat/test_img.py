from PIL import Image
import torch
img = Image.open("datasets/house3k/view_000.png").convert("RGB")
tensor_img = transforms.ToTensor()(img)
print(tensor_img.shape)  # 確保回傳的是 (3, H, W)
