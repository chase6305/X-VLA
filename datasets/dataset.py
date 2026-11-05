# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Iterable, List
import os
import io, json, random, numpy as np, torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from mmengine import fileio
from .utils import action_slice
from .domain_config import DATA_WEIGHTS, DATA_DOMAIN_ID
from .domain_handler.registry import get_handler_cls

class InfiniteDataReader(IterableDataset):
    """
    Output sample:
      {
        'domain_id': LongTensor[],    # domain id
        'language_instruction': str,
        'image_input': FloatTensor[V, C, H, W],
        'image_mask': BoolTensor[V],
        'proprio': FloatTensor[dim_proprio],
        'action': FloatTensor[T, dim_action]
      }
    """
    def __init__(self, 
                 metas_path: str, 
                 num_actions: int = 10, 
                 num_views: int = 3, 
                 training: bool = True,
                 action_mode: str = "ee6d",
                 lang_aug: str = None,
                 ):
        for name, value in (("num_actions", num_actions), ("num_views", num_views)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.num_views = num_views
        self.training = training
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.metas: Dict[str, dict] = {}
        print("use action mode:", action_mode)
        if fileio.isdir(metas_path):
            meta_files = list(fileio.list_dir_or_file(metas_path, suffix=".json", recursive=True, list_dir=False))
            root = metas_path
        else: meta_files, root = [metas_path], ""
        
        for file in meta_files:
            file_path = fileio.join_path(root, file)
            with io.BytesIO(fileio.get(file_path)) as f: meta = json.load(f)
            ### General Style
            if 'dataset_name' in meta.keys() and 'datalist' in meta.keys():
                print(f"== dataset {meta['dataset_name']} with {len(meta['datalist'])} trajs")
                dataset_key = meta["dataset_name"]
            ### Lerobot v2.1 style
            elif "codebase_version" in meta.keys() and meta["codebase_version"] == 'v2.1':
                meta['datalist'] = []
                if "root_path" not in meta.keys(): meta['root_path'] = "/".join(file_path.split("/")[:-2])
                episodes_path = fileio.join_path("/".join(file_path.split("/")[:-1]), "episodes.jsonl")
                with (open(episodes_path, "rb")
                      if "://" not in episodes_path and os.path.isfile(episodes_path)
                      else io.BytesIO(fileio.get(episodes_path))) as f:
                    for line in f: meta['datalist'].append(json.loads(line.decode("utf-8")))
                dataset_key = meta['root_path']
                print(f"== lerobot dataset {meta['robot_type']} with {meta['total_episodes']} trajs at {meta['root_path']}====")
            else: raise NotImplementedError(f"unrecognized meta file format: {file}")
            if dataset_key in self.metas:
                raise ValueError(f"Duplicate dataset identifier {dataset_key!r} in {file_path!r}. "
                                 "Use distinct dataset identifiers to avoid replacing loaded trajectories.")
            get_handler_cls(meta.get("robot_type", dataset_key))
            self.metas[dataset_key] = meta

        if training and not self.metas:
            raise ValueError(f"No dataset metadata found in {metas_path!r} for training.")

        self.image_aug = [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.) \
                if training else torch.nn.Identity(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
        ]
        self.image_aug = transforms.Compose(self.image_aug)

    def _iter_one_dataset(self, dataset_name: str, trajectory_offset: int = 0) -> Iterable[dict]:
        meta = self.metas[dataset_name]
        if 'robot_type' in meta.keys(): robot_type = meta['robot_type']
        else: robot_type = dataset_name
        Handler = get_handler_cls(robot_type)

        while True:
            traj_indices: List[int] | range = range(len(meta["datalist"]))
            if self.training:
                traj_indices = list(traj_indices)
                random.shuffle(traj_indices)
            else:
                worker = get_worker_info()
                if worker is not None:
                    start = (worker.id - trajectory_offset) % worker.num_workers
                    traj_indices = traj_indices[start::worker.num_workers]
            handler = Handler(meta=meta, num_views=self.num_views)
            produced_sample = False
            for traj_idx in traj_indices:
                for sample in handler.iter_episode(
                    traj_idx,
                    num_actions=self.num_actions,
                    training=self.training,
                    image_aug=self.image_aug,
                    lang_aug_map= meta["lang_aug_map"] if "lang_aug_map" in meta.keys() else None,
                    action_mode = self.action_mode
                ):
                    sample["domain_id"] = torch.tensor(DATA_DOMAIN_ID.get(robot_type, 0))
                    idx_for_delta = sample.pop("idx_for_delta", [])
                    idx_for_mask_proprio = sample.pop("idx_for_mask_proprio", [])
                    sample.update(action_slice(sample.pop("abs_trajectory", None), idx_for_delta, idx_for_mask_proprio))
                    produced_sample = True
                    yield sample
            if not self.training:
                return
            if not produced_sample:
                raise RuntimeError(
                    f"Dataset {dataset_name!r} produced no valid samples in a complete pass "
                    f"over {len(traj_indices)} trajectories. Check episode lengths and filtering."
                )


    def __iter__(self):
        names = list(self.metas.keys())
        if not self.training: 
            trajectory_offset = 0
            for n in names:
                yield from self._iter_one_dataset(n, trajectory_offset=trajectory_offset)
                trajectory_offset += len(self.metas[n]['datalist'])
        else:
            #names = names * 2 # increase the dataset sampling frequency
            gens = [iter(self._iter_one_dataset(n)) for n in names]
            ws = [DATA_WEIGHTS.get(n, 1.0) for n in names]
            s = sum(ws); ws = [w / s for w in ws]
            while True:
                i = random.choices(range(len(names)), weights=ws, k=1)[0]
                yield next(gens[i])
