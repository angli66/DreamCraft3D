import bisect
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


@dataclass
class MultiImageDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    n_ref_views: int = 2
    pose_perturb: bool = False
    images_path: str = ""
    image_path: str = ""
    height: Any = 96
    width: Any = 96
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    default_elevation_degs: List[float] = field(default_factory=lambda: [])
    default_elevation_deg: float = 0.0
    default_azimuth_degs: List[float] = field(default_factory=lambda: [])
    default_azimuth_deg: float = 0.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    requires_depth: bool = False
    requires_normal: bool = False
    rays_d_normalize: bool = True
    use_mixed_camera_config: bool = False


class MultiImageDataBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: MultiImageDataModuleConfig = cfg

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            # FIXME:
            if self.cfg.use_mixed_camera_config:
                if self.rank % 2 == 0:
                    random_camera_cfg.camera_distance_range = [
                        self.cfg.default_camera_distance,
                        self.cfg.default_camera_distance,
                    ]
                    random_camera_cfg.fovy_range = [
                        self.cfg.default_fovy_deg,
                        self.cfg.default_fovy_deg,
                    ]
                    self.fixed_camera_intrinsic = True
                else:
                    self.fixed_camera_intrinsic = False
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )

        elevation_deg = torch.FloatTensor(self.cfg.default_elevation_degs)
        azimuth_deg = torch.FloatTensor(self.cfg.default_azimuth_degs)
        camera_distance = torch.FloatTensor([self.cfg.default_camera_distance])

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        camera_position: Float[Tensor, "self.cfg.n_ref_views 3"] = torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation),
            ],
            dim=-1,
        )

        center: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 0], dtype=torch.float32)[
            None
        ]
        center: Float[Tensor, "self.cfg.n_ref_views 3"] = center.repeat(
            self.cfg.n_ref_views, 1
        )
        up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]
        up: Float[Tensor, "self.cfg.n_ref_views 3"] = up.repeat(self.cfg.n_ref_views, 1)

        light_position: Float[Tensor, "self.cfg.n_ref_views 3"] = camera_position
        lookat: Float[Tensor, "self.cfg.n_ref_views 3"] = F.normalize(
            center - camera_position, dim=-1
        )
        right: Float[Tensor, "self.cfg.n_ref_views 3"] = F.normalize(
            torch.cross(lookat, up), dim=-1
        )
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        self.c2w: Float[Tensor, "self.cfg.n_ref_views 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )
        self.c2w4x4: Float[Tensor, "self.cfg.n_ref_views 4 4"] = torch.cat(
            [self.c2w, torch.zeros_like(self.c2w[:, :1])], dim=1
        )
        self.c2w4x4[:, 3, 3] = 1.0

        self.camera_position = camera_position
        self.light_position = light_position
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distance = camera_distance
        self.fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.focal_lengths = [
            0.5 * height / torch.tan(0.5 * self.fovy) for height in self.heights
        ]

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.focal_length = self.focal_lengths[0]
        self.set_rays()
        self.load_images()
        self.prev_height = self.height

    def set_rays(self):
        # get directions by dividing directions_unit_focal by focal length
        directions: Float[Tensor, "1 H W 3"] = self.directions_unit_focal[None]
        directions = directions.repeat(self.cfg.n_ref_views, 1, 1, 1)
        directions[:, :, :, :2] = directions[:, :, :, :2] / self.focal_length
        self.directions = directions

        rays_o, rays_d = get_rays(
            directions,
            self.c2w,
            keepdim=True,
            noise_scale=self.cfg.rays_noise_scale,
            normalize=self.cfg.rays_d_normalize,
        )

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.01, 100.0
        )  # FIXME: hard-coded near and far
        proj_mtx = proj_mtx.repeat(self.cfg.n_ref_views, 1, 1)
        mvp_mtx: Float[Tensor, "self.cfg.n_ref_views 4 4"] = get_mvp_matrix(
            self.c2w, proj_mtx
        )

        self.proj_mtx = proj_mtx
        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx

    def load_images(self):
        # load images
        self.rgb = torch.zeros(
            self.cfg.n_ref_views, self.height, self.width, 3, dtype=torch.float32
        ).to(self.rank)
        self.mask = torch.zeros(
            self.cfg.n_ref_views, self.height, self.width, 1, dtype=torch.bool
        ).to(self.rank)
        if self.cfg.requires_depth:
            self.depth = torch.zeros(
                self.cfg.n_ref_views, self.height, self.width, dtype=torch.float32
            ).to(self.rank)
        else:
            self.depth = None
        if self.cfg.requires_normal:
            self.normal = torch.zeros(
                self.cfg.n_ref_views, self.height, self.width, 3, dtype=torch.float32
            ).to(self.rank)
        else:
            self.normal = None

        for i in range(self.cfg.n_ref_views):
            assert os.path.exists(
                self.cfg.images_path
            ), f"Could not find images path {self.cfg.images_path}!"

            image_path = os.path.join(self.cfg.images_path, f"{i}_rgba.png")

            rgba = cv2.cvtColor(
                cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
            )
            rgba = (
                cv2.resize(
                    rgba, (self.width, self.height), interpolation=cv2.INTER_AREA
                ).astype(np.float32)
                / 255.0
            )
            rgb = rgba[..., :3]
            self.rgb[i] = torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank)
            self.mask[i] = (
                torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(self.rank)
            )
            print(
                f"[INFO] multi image dataset: load image {image_path} {self.rgb[i].shape}"
            )

            # load depth
            if self.cfg.requires_depth:
                depth_path = image_path.replace("_rgba.png", "_depth.png")
                assert os.path.exists(depth_path)
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                depth = cv2.resize(
                    depth, (self.width, self.height), interpolation=cv2.INTER_AREA
                )
                self.depth[i] = (
                    torch.from_numpy(depth.astype(np.float32) / 255.0)
                    .unsqueeze(0)
                    .to(self.rank)
                )
                print(
                    f"[INFO] multi image dataset: load depth {depth_path} {self.depth[i].shape}"
                )

            # load normal
            if self.cfg.requires_normal:
                normal_path = image_path.replace("_rgba.png", "_normal.png")
                assert os.path.exists(normal_path)
                normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
                normal = cv2.resize(
                    normal, (self.width, self.height), interpolation=cv2.INTER_AREA
                )
                self.normal[i] = (
                    torch.from_numpy(normal.astype(np.float32) / 255.0)
                    .unsqueeze(0)
                    .to(self.rank)
                )
                print(
                    f"[INFO] multi image dataset: load normal {normal_path} {self.normal[i].shape}"
                )

    def get_all_images(self):
        return self.rgb

    def update_step_(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        if self.height == self.prev_height:
            return

        self.prev_height = self.height
        self.width = self.widths[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        self.focal_length = self.focal_lengths[size_ind]
        threestudio.debug(f"Training height: {self.height}, width: {self.width}")
        self.set_rays()
        self.load_images()


class MultiImageIterableDataset(IterableDataset, MultiImageDataBase, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def collate(self, batch) -> Dict[str, Any]:
        # Randomly select a value from self.cfg.n_ref_views and use it as the reference view
        ref_view = torch.randint(self.cfg.n_ref_views, (1,))
        if not self.cfg.pose_perturb:
            batch = {
                "rays_o": self.rays_o[ref_view],
                "rays_d": self.rays_d[ref_view],
                "mvp_mtx": self.mvp_mtx[ref_view],
                "camera_positions": self.camera_position[ref_view],
                "light_positions": self.light_position[ref_view],
                "elevation": self.elevation_deg[ref_view],
                "azimuth": self.azimuth_deg[ref_view],
                "camera_distances": self.camera_distance,
                "rgb": self.rgb[ref_view],
                "ref_depth": self.depth[ref_view] if self.depth is not None else None,
                "ref_normal": self.normal[ref_view] if self.normal is not None else None,
                "mask": self.mask[ref_view],
                "height": self.cfg.height,
                "width": self.cfg.width,
                "c2w": self.c2w[ref_view],
                "c2w4x4": self.c2w4x4[ref_view],
            }
        else:
            # Perturb azimuth, elevation, and camera distance
            azimuth_deg = self.azimuth_deg[ref_view] + (torch.randn(1) - 0.5) * 20
            elevation_deg = self.elevation_deg[ref_view] + (torch.randn(1) - 0.5) * 20
            camera_distance = self.camera_distance + (torch.randn(1) - 0.5) * 0.4

            # Recalculate remaining camera parameters
            elevation = elevation_deg * math.pi / 180
            azimuth = azimuth_deg * math.pi / 180
            camera_position = torch.stack(
                [
                    camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distance * torch.sin(elevation),
                ],
                dim=-1,
            )
            center = torch.as_tensor([0, 0, 0], dtype=torch.float32)[None]
            up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]
            light_position = camera_position
            lookat = F.normalize(center - camera_position, dim=-1)
            right = F.normalize(torch.cross(lookat, up), dim=-1)
            up = F.normalize(torch.cross(right, lookat), dim=-1)
            c2w = torch.cat(
                [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
                dim=-1,
            )
            c2w4x4 = torch.cat([c2w, torch.zeros_like(c2w[:, :1])], dim=1)
            c2w4x4[:, 3, 3] = 1.0

            # Get rays
            rays_o, rays_d = get_rays(
                self.directions[ref_view],
                c2w,
                keepdim=True,
                noise_scale=self.cfg.rays_noise_scale,
                normalize=self.cfg.rays_d_normalize,
            )
            proj_mtx = get_projection_matrix(
                self.fovy, self.width / self.height, 0.01, 100.0
            )
            mvp_mtx = get_mvp_matrix(c2w, proj_mtx)

            batch = {
                "rays_o": rays_o,
                "rays_d": rays_d,
                "mvp_mtx": mvp_mtx,
                "camera_positions": camera_position,
                "light_positions": light_position,
                "elevation": elevation_deg,
                "azimuth": azimuth_deg,
                "camera_distances": camera_distance,
                "rgb": self.rgb[ref_view],
                "ref_depth": self.depth[ref_view] if self.depth is not None else None,
                "ref_normal": self.normal[ref_view] if self.normal is not None else None,
                "mask": self.mask[ref_view],
                "height": self.cfg.height,
                "width": self.cfg.width,
                "c2w": c2w,
                "c2w4x4": c2w4x4,
            }

        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        return batch

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.update_step_(epoch, global_step, on_load_weights)
        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}


class MultiImageDataset(Dataset, MultiImageDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        if self.split == "val":
            return self.cfg.n_ref_views
        else:
            return len(self.random_pose_generator)

    def __getitem__(self, index):
        if self.split == "val":
            ref_view = torch.tensor([index])
            batch = {
                "index": index,
                "rays_o": self.rays_o[index],
                "rays_d": self.rays_d[index],
                "mvp_mtx": self.mvp_mtx[index],
                "c2w": self.c2w4x4[index],
                "camera_positions": self.camera_position[index],
                "light_positions": self.light_position[index],
                "elevation": self.elevation_deg[index],
                "azimuth": self.azimuth_deg[index],
                "camera_distances": self.camera_distance,
                "height": self.height,
                "width": self.width,
                "fovy": self.fovy,
                "proj_mtx": self.proj_mtx[index],
                "mvp_mtx_ref": self.mvp_mtx[ref_view].squeeze(0),
                "c2w_ref": self.c2w4x4[ref_view],
            }
        else:
            # Randomly select a value from self.cfg.n_ref_views and use it as the reference view
            ref_view = torch.tensor([0])
            batch = self.random_pose_generator[index]
            batch.update(
                {
                    "height": self.random_pose_generator.cfg.eval_height,
                    "width": self.random_pose_generator.cfg.eval_width,
                    "mvp_mtx_ref": self.mvp_mtx[ref_view].squeeze(0),
                    "c2w_ref": self.c2w4x4[ref_view],
                }
            )
        return batch


@register("multi-image-datamodule")
class MultiImageDataModule(pl.LightningDataModule):
    cfg: MultiImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiImageDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MultiImageIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MultiImageDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MultiImageDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
