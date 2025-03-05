import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset


from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetHouse3KCfg(DatasetCfgCommon):
    name: Literal["house3k"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_times_per_scene: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True


class DatasetHouse3K(Dataset):
    cfg: DatasetHouse3KCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    # chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetHouse3KCfg,
        stage: Stage,
        # view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        # self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        '''
        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # NOTE: hack to skip some chunks in testing during training, but the index
            # is not change, this should not cause any problem except for the display
            self.chunks = self.chunks[:: cfg.test_chunk_interval]
        '''
        self.camera_data = self.load_camera_data()
        self.data_index = list(self.camera_data.keys())

    '''
    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # print(chunk_path)
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            # for example in chunk:
            times_per_scene = self.cfg.test_times_per_scene
            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]

                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                if times_per_scene > 1:  # specifically for DTU
                    scene = f"{example['key']}_{(run_idx % times_per_scene):02d}"
                else:
                    scene = example["key"]

                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                    # reverse the context
                    # context_indices = torch.flip(context_indices, dims=[0])
                    # print(context_indices)
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                # Load the images.
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_images = self.convert_images(context_images)
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images)

                # Skip the example if the images don't have the right shape.
                context_image_invalid = context_images.shape[1:] != (3, 360, 640)
                target_image_invalid = target_images.shape[1:] != (3, 360, 640)
                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue

                # Resize the world to make the baseline 1.
                context_extrinsics = extrinsics[context_indices]
                if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                    a, b = context_extrinsics[:, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_epsilon:
                        print(
                            f"Skipped {scene} because of insufficient baseline "
                            f"{scale:.6f}"
                        )
                        continue
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1

                nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
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
                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))
'''
    def load_camera_data(self) -> dict:
        """加载 camera.json 并解析数据"""
        merged_data = {}
        for root in self.cfg.roots:
            camera_json_path = root / self.stage / "camera_new.json"
            if camera_json_path.exists():
                with camera_json_path.open("r") as f:
                    camera_data = json.load(f)
                # 合并数据，key 是 frame ID，value 是相机参数
                merged_data.update(camera_data)
        return merged_data
    
    def __len__(self) -> int:
        return min(len(self.data_index) * self.cfg.test_times_per_scene, self.cfg.test_len) if self.stage == "test" and self.cfg.test_len > 0 else len(self.data_list) * self.cfg.test_times_per_scene

    def __getitem__(self, idx: int):
        """根据索引返回数据"""
        frame_id = self.data_index[idx // self.cfg.test_times_per_scene]
        camera_info = self.camera_data[frame_id]

        # 解析相机参数
        intrinsics = self.convert_intrinsics(camera_info["intrinsics"])
        extrinsics = self.convert_extrinsics(camera_info["extrinsics"])
        near = torch.tensor(camera_info["near"], dtype=torch.float32)
        far = torch.tensor(camera_info["far"], dtype=torch.float32)

        # 读取图像
        image_path = Path(self.cfg.roots[0]) / self.stage / camera_info["image_path"]
        image = self.to_tensor(Image.open(image_path))

        # 组织返回数据
        example = {
            "context": {
                "extrinsics": extrinsics.unsqueeze(0),
                "intrinsics": intrinsics.unsqueeze(0),
                "image": image.unsqueeze(0),
                "near": self.get_bound("near", 1),
                "far": self.get_bound("far", 1),
                "index": frame_id,
            },
            "target": {
                "extrinsics": extrinsics.unsqueeze(0),
                "intrinsics": intrinsics.unsqueeze(0),
                "image": image.unsqueeze(0),
                "near": self.get_bound("near", 1),
                "far": self.get_bound("near", 1),
                "index": frame_id,
            },
            "scene": frame_id,
        }

        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        return apply_crop_shim(example, tuple(self.cfg.image_shape))
    '''
    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics
    '''
    def convert_intrinsics(self, intrinsics: list[float]) -> Tensor:
        K = torch.eye(3, dtype=torch.float32)
        K[0, 0] = intrinsics[0]  # fx
        K[1, 1] = intrinsics[1]  # fy
        K[0, 2] = intrinsics[2]  # cx
        K[1, 2] = intrinsics[3]  # cy
        return K
    '''
    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
    '''
    
    def convert_extrinsics(self, extrinsics: list[float]) -> Tensor:
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32).reshape(3, 4)
        E = torch.eye(4, dtype=torch.float32)
        E[:3, :4] = extrinsics
        return E.inverse()
    
    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)