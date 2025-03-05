import torch

# 讀取 .torch 文件
torch_file_path = "datasets/re10k/test/000000.torch"  # 你的 .torch 檔案路徑
data = torch.load(torch_file_path)

# 顯示 .torch 檔案的場景數量
print(f"✅ 成功載入 `{torch_file_path}`，共包含 {len(data)} 個場景！")

# 解析第一個場景
first_scene = data[0]
print("\n📂 第一個場景的結構:")
for key, value in first_scene.items():
    if isinstance(value, torch.Tensor):
        print(f" - {key}: Tensor (shape: {value.shape})")
    elif isinstance(value, list):
        print(f" - {key}: List (length: {len(value)})")
    else:
        print(f" - {key}: {value}")

# # 顯示相機參數的詳細數值（前 5 組）
# if "cameras" in first_scene:
#     print("\n📸 第一個場景的相機參數（前 5 組數值）:")
#     print(first_scene["cameras"][:5])

print(data[0])