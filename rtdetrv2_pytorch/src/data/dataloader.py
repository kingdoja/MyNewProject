"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import default_collate

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode

import random
from functools import partial

from ..core import register


__all__ = [
    'DataLoader',
    'BaseCollateFunction',
    'BatchImageCollateFuncion',
    'batch_image_collate_fn',
    'CocoImageWeightedRandomSampler',
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn', 'sampler']

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=None,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor=None,
        persistent_workers=False,
        pin_memory_device="",
    ):
        if sampler is not None and hasattr(sampler, 'bind_dataset'):
            sampler.bind_dataset(dataset)
        if sampler is not None and shuffle:
            warnings.warn("`shuffle` is ignored because a custom sampler is provided.", RuntimeWarning)
            shuffle = False
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch 
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)
    
    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch 

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


@register()
class BatchImageCollateFuncion(BaseCollateFunction):
    def __init__(
        self, 
        scales=None, 
        stop_epoch=None, 
    ) -> None:
        super().__init__()
        self.scales = scales
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        # self.interpolation = interpolation

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            # sz = random.choice(self.scales)
            # sz = [sz] if isinstance(sz, int) else list(sz)
            # VF.resize(inpt, sz, interpolation=self.interpolation)

            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'], size=sz, mode='nearest')
                raise NotImplementedError('')

        return images, targets


@register()
class CocoImageWeightedRandomSampler(data.Sampler):
    """Weighted sampler aligned with COCO-style datasets."""

    __inject__ = ['dataset']

    def __init__(
        self,
        weights_file: str,
        dataset=None,
        num_samples: int | None = None,
        replacement: bool = True,
        default_weight: float = 1.0,
    ) -> None:
        self.weights_file = weights_file
        self.num_samples = num_samples
        self.replacement = replacement
        self.default_weight = default_weight
        self._sampler = None
        if dataset is not None:
            self.bind_dataset(dataset)

    def bind_dataset(self, dataset) -> None:
        dataset_ids = self._extract_dataset_ids(dataset)
        if dataset_ids is None:
            raise AttributeError(
                "Dataset must expose COCO image ids via `ids` or `coco` for weighted sampling."
            )
        mapping = self._load_weights(self.weights_file)
        weights = self._build_weight_vector(dataset_ids, mapping, self.default_weight)
        tensor = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = self.num_samples or len(weights)
        self._sampler = data.WeightedRandomSampler(tensor, self.num_samples, replacement=self.replacement)

    @staticmethod
    def _extract_dataset_ids(dataset):
        # Common path: torchvision/faster_coco_eval datasets expose `ids`.
        if hasattr(dataset, 'ids'):
            return list(dataset.ids)

        # Fallback for wrapped datasets that still hold a COCO api object.
        coco = getattr(dataset, 'coco', None)
        if coco is not None:
            if hasattr(coco, 'getImgIds'):
                return list(coco.getImgIds())
            if hasattr(coco, 'imgs'):
                return list(coco.imgs.keys())

        # Fallback for torch.utils.data.Subset-like wrappers.
        base = getattr(dataset, 'dataset', None)
        indices = getattr(dataset, 'indices', None)
        if base is not None and indices is not None:
            base_ids = CocoImageWeightedRandomSampler._extract_dataset_ids(base)
            if base_ids is not None:
                return [base_ids[i] for i in indices]

        return None

    @staticmethod
    def _load_weights(weights_file: str):
        path = Path(weights_file)
        if not path.exists():
            raise FileNotFoundError(f"weights_file {path} not found")
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
        return payload.get('image_weights', {})

    @staticmethod
    def _build_weight_vector(dataset_ids, mapping, default_weight: float):
        weights = []
        for image_id in dataset_ids:
            key = str(image_id)
            info = mapping.get(key)
            if info is None and isinstance(image_id, (int, str)):
                info = mapping.get(int(image_id))
            weights.append(float(info.get('weight', default_weight)) if info else default_weight)
        return weights

    def __iter__(self):
        if self._sampler is None:
            raise RuntimeError("CocoImageWeightedRandomSampler has not been bound to a dataset.")
        return iter(self._sampler)

    def __len__(self):
        if self.num_samples is None:
            raise RuntimeError("CocoImageWeightedRandomSampler has not been initialized.")
        return self.num_samples

