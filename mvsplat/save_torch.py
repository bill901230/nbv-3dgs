import torch
import json

# 讀取 .torch 文件
torch_file_path = "datasets/re10k/train/000001.torch"  # 更新為你的文件路徑
data = torch.load(torch_file_path)

# 轉換 Tensor 為可序列化格式
def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, list):
        return [tensor_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: tensor_to_list(value) for key, value in obj.items()}
    else:
        return obj

serializable_data = tensor_to_list(data[0])

# 存成 JSON 文件
json_file_path = "datasets/re10k/train/000001_torch_data.json"
with open(json_file_path, "w") as json_file:
    json.dump(serializable_data, json_file, indent=4)

print(f"✅ .torch 內容已轉換並存為 {json_file_path}")
