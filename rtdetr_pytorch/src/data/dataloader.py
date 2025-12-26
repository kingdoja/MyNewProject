from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch
import torch.utils.data as data

from src.core import register


__all__ = ['DataLoader', 'CocoImageWeightedRandomSampler']


@register
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



@register
def default_collate_fn(items):
    '''default collate_fn
    '''    
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


@register
class CocoImageWeightedRandomSampler(data.Sampler):
    """Weighted sampler that aligns JSON image weights with the current dataset order."""

    __inject__ = ['dataset']

    def __init__(
        self,
        dataset,
        weights_file: str,
        num_samples: int | None = None,
        replacement: bool = True,
        default_weight: float = 1.0,
    ):
        if not hasattr(dataset, 'ids'):
            raise AttributeError("Dataset must expose `ids` compatible with COCO image ids.")
        mapping = self._load_weights(weights_file)
        weights = self._build_weight_vector(dataset.ids, mapping, default_weight)
        tensor = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples or len(weights)
        self._sampler = data.WeightedRandomSampler(tensor, self.num_samples, replacement=replacement)

    def _load_weights(self, weights_file: str):
        path = Path(weights_file)
        if not path.exists():
            raise FileNotFoundError(f"weights_file {path} not found")
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
        return payload.get('image_weights', {})

    def _build_weight_vector(self, dataset_ids, mapping, default_weight: float):
        weights = []
        for image_id in dataset_ids:
            key = str(image_id)
            info = mapping.get(key) or mapping.get(int(image_id)) if isinstance(image_id, (int, str)) else None
            weights.append(float(info.get('weight', default_weight)) if info else default_weight)
        return weights

    def __iter__(self):
        return iter(self._sampler)

    def __len__(self):
        return self.num_samples
