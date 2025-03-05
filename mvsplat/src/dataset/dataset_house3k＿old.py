import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torchvision import transforms
from PIL import Image
from functools import cached_property
from typing import Literal
from dataclasses import dataclass
from pathlib import Path

from .dataset import DatasetCfgCommon
from .types import Stage

@dataclass
class DatasetHouse3KCfg(DatasetCfgCommon):
    name: Literal["house3k"]  # 數據集名稱
    roots: list[Path]  # 數據集路徑
    augment: bool  # 是否進行數據增強
    test_len: int  # 測試數據大小
    test_times_per_scene: int  # 每個場景測試的次數
    skip_bad_shape: bool = True  # 是否跳過異常影像
    shuffle_val: bool = False  # 是否在驗證時打亂數據

class DatasetHouse3K(Dataset):
    cfg: DatasetHouse3KCfg
    stage: Stage

    def __init__(self, cfg: DatasetHouse3KCfg, stage: Stage):
        super().__init__()
        self.cfg = cfg  # 儲存設定
        self.stage = stage  # 設定 train / test / val 模式
        self.transform = transforms.ToTensor()

        # 讀取 cameras.json
        with open(cfg.roots[0] / "camera.json", "r") as f:
            self.cameras = json.load(f)

    @cached_property
    def index(self) -> dict[str, dict]:
        """讀取 `evaluation_index_house3k.json` 作為數據索引"""
        with open(self.cfg.roots[0] / "../../assets/evaluation_index_house3k.json", "r") as f:
            data = json.load(f)
        return {str(k): v for k, v in data.items()}  # 回傳 {scene_id: {"context": [...], "target": [...]}}

    @property
    def data_stage(self) -> str:
        """根據當前模式決定數據類別"""
        return "test" if self.stage in ["test", "val"] else self.stage

    def load_camera_params(self, view):
        """讀取相機 `intrinsics` 和 `extrinsics`"""
        cam_data = self.cameras[str(view)]
        intrinsics = torch.tensor(cam_data["intrinsics"], dtype=torch.float32)
        extrinsics = torch.tensor(cam_data["extrinsics"], dtype=torch.float32)
        return intrinsics, extrinsics

    def compute_near_far(self, extrinsics):
        """根據 `extrinsics` 計算 `near` 和 `far`"""
        camera_positions = extrinsics[:, :3, 3]  # 取得相機位置
        distances = torch.norm(camera_positions, dim=1)  # 計算到原點的距離
        near = torch.min(distances) * 0.9  # 取最小距離的 90%
        far = torch.max(distances) * 1.1  # 取最大距離的 110%
        return near, far

    def get_bound(self, bound: Literal["near", "far"], extrinsics) -> torch.Tensor:
        """取得 `near` 和 `far`，適應 `context` & `target`"""
        near, far = self.compute_near_far(extrinsics)
        return torch.full((extrinsics.shape[0],), near if bound == "near" else far, dtype=torch.float32)

    def load_image(self, image_path):
        """讀取影像並轉換為 PyTorch Tensor"""
        img = Image.open(self.cfg.roots[0] / image_path).convert("RGB")
        return self.transform(img)  # 轉換為 (3, H, W) Tensor

    def __getitem__(self, index):
        scene_keys = list(self.index.keys())
        scene_id = scene_keys[index] 
        if scene_id not in self.index:
            raise KeyError(f"scene_id {scene_id} 不存在於 `evaluation_index_house3k.json`！")


        scene_data = self.index[scene_id]

        context_indices = scene_data["context"]
        target_indices = scene_data["target"]

        def load_data(views):
            images, intrinsics, extrinsics, indices = [], [], [], []
            for view in views:
                cam_data = self.cameras[str(view)]
                img = self.load_image(cam_data["image_path"])
                # if img is None or img.sum() == 0:
                #     print(f"警告：影像 {cam_data['image_path']} 加載失敗！")
                # else: print(f"{img.sum()}")
                # images.append(self.load_image(cam_data["image_path"]))
                images.append(img)
                intrinsics.append(torch.tensor(cam_data["intrinsics"], dtype=torch.float32))
                extrinsics.append(torch.tensor(cam_data["extrinsics"], dtype=torch.float32))
                indices.append(view)
            return torch.stack(images), torch.stack(intrinsics), torch.stack(extrinsics), torch.tensor(indices, dtype=torch.long)

        context_images, context_intrinsics, context_extrinsics, context_indices = load_data(context_indices)
        target_images, target_intrinsics, target_extrinsics, target_indices = load_data(target_indices)

        context_near = self.get_bound("near", context_extrinsics)
        context_far = self.get_bound("far", context_extrinsics)
        target_near = self.get_bound("near", target_extrinsics)
        target_far = self.get_bound("far", target_extrinsics)

        return {
            "context": {
                "extrinsics": context_extrinsics,
                "intrinsics": context_intrinsics,
                "image": context_images,
                "near": context_near,
                "far": context_far,
                "index": context_indices,
            },
            "target": {
                "extrinsics": target_extrinsics,
                "intrinsics": target_intrinsics,
                "image": target_images,
                "near": target_near,
                "far": target_far,
                "index": target_indices,
            },
            "scene": scene_id,
        }

    def __len__(self) -> int:
        """計算測試數據的長度"""
        # return (
        #     min(len(self.index.keys()) * self.cfg.test_times_per_scene, self.cfg.test_len)
        #     if self.stage == "test" and self.cfg.test_len > 0
        #     else len(self.index.keys()) * self.cfg.test_times_per_scene
        # )
        dataset_length = len(self.index.keys())  # 直接取 JSON 的鍵數量
        print(f"⚠️ DEBUG: Dataset Length={dataset_length}")  # 記錄長度
        return dataset_length
