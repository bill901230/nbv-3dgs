import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class DatasetHouse3K(Dataset):
    def __init__(self, data_root, split='train', transform=None):
        """
        Args:
            data_root (str): House3K 數據集根目錄。
            split (str): 選擇數據集的拆分（train/val/test）。
            transform: 可選的圖像變換。
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform if transform else ToTensor()
        
        # 讀取 evaluation_index_house3k.json
        eval_index_path = os.path.join(data_root, 'evaluation_index_house3k.json')
        with open(eval_index_path, 'r') as f:
            self.eval_data = json.load(f)
        
        # 讀取 cameras.json
        cameras_path = os.path.join(data_root, 'cameras.json')
        with open(cameras_path, 'r') as f:
            self.cameras = json.load(f)
        
        # 獲取所有場景
        self.scenes = list(self.eval_data.keys())

    def get_bound(self, bound_type, num_samples):
        """獲取 near 和 far 界限，這裡假設固定範圍，實際可根據數據計算"""
        bounds = {'near': 0.1, 'far': 10.0}  # 假設的 near 和 far 范圍
        return torch.full((num_samples,), bounds[bound_type], dtype=torch.float32)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        context_indices = self.eval_data[scene]['context']
        target_indices = self.eval_data[scene]['target']
        
        # 讀取內外參數
        extrinsics = torch.tensor(self.cameras['extrinsics'], dtype=torch.float32)
        intrinsics = torch.tensor(self.cameras['intrinsics'], dtype=torch.float32)
        
        # 讀取 context 圖像
        context_images = [self.load_image(scene, i) for i in context_indices]
        context_images = torch.stack(context_images)
        
        # 讀取 target 圖像
        target_images = [self.load_image(scene, i) for i in target_indices]
        target_images = torch.stack(target_images)
        
        # Normalization factor
        nf_scale = 1.0
        
        # 構造輸出格式
        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": self.get_bound("near", len(context_indices)) / nf_scale,
                "far": self.get_bound("far", len(context_indices)) / nf_scale,
                "index": context_indices,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": self.get_bound("near", len(target_indices)) / nf_scale,
                "far": self.get_bound("far", len(target_indices)) / nf_scale,
                "index": target_indices,
            },
            "scene": scene,
        }
        
        return example
    
    def load_image(self, scene, index):
        """從磁碟加載影像，並轉換為 Tensor"""
        img_path = os.path.join(self.data_root, 'images', scene, f'{index:06d}.png')
        image = torch.from_numpy(np.array(Image.open(img_path).convert('RGB')))  # PIL -> NumPy -> Tensor
        return self.transform(image)
