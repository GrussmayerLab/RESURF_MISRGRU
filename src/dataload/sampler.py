"""
samplers.py

Batch samplers that use Dataset grouping information.

This module expects a dataset that exposes:
    dataset.pattern_to_indices : Dict[str, List[int]]

In this repo, GroupedPatternDataset builds that mapping so we can sample batches
that contain multiple pattern IDs (diversity per batch) 
and the # different patterns per batch can be defined.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence


class FixedPatternBatchSampler:
    """
    Create batches by selecting a random number of patterns (between min_patterns and max_patterns)
    and drawing approximately equal numbers of samples from each selected pattern.

    Requirements
    -----------
    The dataset must provide:
        dataset.pattern_to_indices: dict {pattern_id -> list of sample indices}

    Notes
    -----
    - This sampler aims to use each sample index at most once per epoch (no replacement).
    - If there are not enough remaining patterns to satisfy `min_patterns`, iteration stops.
    - Batches that cannot reach full `batch_size` are discarded.
    - If `shuffle_each_epoch=True`, batches are regenerated every time you iterate.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 10,
        min_patterns: int = 2,
        max_patterns: int = 2,
        *,
        shuffle_each_epoch: bool = False,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.min_patterns = int(min_patterns)
        self.max_patterns = int(max_patterns)
        self.shuffle_each_epoch = bool(shuffle_each_epoch)
        self.seed = seed

        # Validate
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.min_patterns <= 0:
            raise ValueError("min_patterns must be > 0")
        if self.max_patterns < self.min_patterns:
            raise ValueError("max_patterns must be >= min_patterns")

        if not hasattr(dataset, "pattern_to_indices"):
            raise AttributeError("Dataset must have attribute 'pattern_to_indices'")

        self.pattern_to_indices: Dict[str, List[int]] = {
            str(pid): list(idxs) for pid, idxs in dataset.pattern_to_indices.items()
        }
        self.pattern_ids: List[str] = sorted(self.pattern_to_indices.keys())

        if len(self.pattern_ids) == 0:
            raise ValueError("Dataset.pattern_to_indices is empty; cannot build pattern-aware batches.")

        # RNG: seeded for reproducibility if requested
        self._rng = random.Random(self.seed)

        # Precompute batches once if we do NOT reshuffle each epoch.
        self._batches: Optional[List[List[int]]] = None
        if not self.shuffle_each_epoch:
            self._batches = self._create_batches()

    def _create_batches(self) -> List[List[int]]:
        """Create a list of batches (each batch is a list of dataset indices)."""
        used = set()
        all_indices = set(range(len(self.dataset)))

        # Copy to avoid mutating dataset mapping
        available_by_pattern: Dict[str, List[int]] = {
            pid: list(idxs) for pid, idxs in self.pattern_to_indices.items()
        }

        # Deterministic starting order for reproducibility, then shuffle with RNG
        for pid in available_by_pattern:
            available_by_pattern[pid].sort()

        batches: List[List[int]] = []

        while used != all_indices:
            # Patterns with at least one unused index
            eligible_patterns = [
                pid for pid, idxs in available_by_pattern.items()
                if any(i not in used for i in idxs)
            ]

            if len(eligible_patterns) < self.min_patterns:
                break  # not enough distinct patterns left

            n_patterns = min(len(eligible_patterns), self._rng.randint(self.min_patterns, self.max_patterns))
            selected_patterns = self._rng.sample(eligible_patterns, n_patterns)

            # Split batch budget across selected patterns as evenly as possible
            items_per_pattern = [self.batch_size // n_patterns] * n_patterns
            for i in range(self.batch_size % n_patterns):
                items_per_pattern[i] += 1

            batch: List[int] = []
            for pid, n_items in zip(selected_patterns, items_per_pattern):
                candidates = [i for i in available_by_pattern[pid] if i not in used]
                if not candidates:
                    continue

                # Sample without replacement from that pattern
                take = min(n_items, len(candidates))
                picked = self._rng.sample(candidates, take)

                batch.extend(picked)
                used.update(picked)

            # Keep only full batches
            if len(batch) == self.batch_size:
                batches.append(batch)

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        # Regenerate batches each epoch if requested
        if self.shuffle_each_epoch:
            self._batches = self._create_batches()

        assert self._batches is not None
        for batch in self._batches:
            yield batch

    def __len__(self) -> int:
        # If we shuffle each epoch, length can vary. We return the current/precomputed value.
        if self._batches is None:
            self._batches = self._create_batches()
        return len(self._batches)
