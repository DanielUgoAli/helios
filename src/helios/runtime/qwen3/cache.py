from collections.abc import Sequence
from dataclasses import dataclass

import torch

from helios.runtime.qwen3.config import Qwen3Config


@dataclass
class LayerKV:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class LayerKVSnapshot:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class KVCacheSnapshot:
    length: int
    layers: tuple[LayerKVSnapshot, ...]


@dataclass(frozen=True)
class KVBlockSnapshot:
    length: int
    layers: tuple[LayerKVSnapshot, ...]


class KVCache:
    """Fixed-size KV storage with one independent logical sequence per row."""

    def __init__(
        self,
        config: Qwen3Config,
        capacity: int,
        *,
        device: torch.device,
        batch_size: int = 1,
    ) -> None:
        if not 1 <= capacity <= config.context_length:
            raise ValueError(
                f"KV-cache capacity must be between 1 and {config.context_length:,} tokens."
            )
        if batch_size < 1:
            raise ValueError("KV-cache batch size must be positive.")
        self.capacity = capacity
        self.batch_size = batch_size
        self._common_length: int | None = 0
        self._slot_lengths = [0] * batch_size
        self._device = device
        self._layers = [
            LayerKV(
                keys=torch.zeros(
                    batch_size,
                    config.n_kv_heads,
                    capacity,
                    config.head_dim,
                    device=device,
                    dtype=config.dtype,
                ),
                values=torch.zeros(
                    batch_size,
                    config.n_kv_heads,
                    capacity,
                    config.head_dim,
                    device=device,
                    dtype=config.dtype,
                ),
            )
            for _ in range(config.n_layers)
        ]

    @property
    def length(self) -> int:
        """The common row length for a single-row cache."""
        if self._common_length is None:
            raise RuntimeError(
                "This KV cache has independent row lengths; use slot_length()."
            )
        return self._common_length

    @property
    def memory_bytes_per_token(self) -> int:
        return sum(
            tensor[:, :, :1].numel() * tensor.element_size()
            for layer in self._layers
            for tensor in (layer.keys, layer.values)
        )

    @property
    def memory_bytes_per_slot_token(self) -> int:
        return self.memory_bytes_per_token // self.batch_size

    def slot_length(self, slot: int) -> int:
        return self._slot_lengths[self._slot(slot)]

    def slot_lengths(self, slots: Sequence[int] | torch.Tensor) -> torch.Tensor:
        rows = self.slot_ids(slots)
        return torch.tensor(
            [self._slot_lengths[slot] for slot in rows],
            dtype=torch.long,
            device=self._device,
        )

    def slot_ids(self, slots: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
        return self._rows(slots)

    def clear_slot(self, slot: int) -> None:
        self._slot_lengths[self._slot(slot)] = 0
        self._common_length = None

    def append(
        self,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        slots: Sequence[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if keys.shape != values.shape:
            raise ValueError("Key and value tensors must have identical shapes.")
        entry = self._layers[layer]
        if (
            keys.ndim != 4
            or keys.shape[1] != entry.keys.shape[1]
            or keys.shape[-1] != entry.keys.shape[-1]
        ):
            raise ValueError("KV-cache tensor shape does not match this Qwen3 model.")
        if slots is None:
            if keys.shape[0] != self.batch_size:
                raise ValueError("KV-cache rows must match the key/value batch size.")
            start = self.length
            end = start + keys.shape[2]
            if end > self.capacity:
                raise ValueError(
                    f"KV cache overflow: capacity is {self.capacity:,} tokens."
                )
            entry.keys[:, :, start:end].copy_(keys)
            entry.values[:, :, start:end].copy_(values)
            return entry.keys[:, :, :end], entry.values[:, :, :end]

        rows = self._rows(slots)
        if keys.shape[0] != len(rows):
            raise ValueError("KV-cache rows must match the key/value batch size.")
        starts = tuple(self._slot_lengths[slot] for slot in rows)
        ends = tuple(start + keys.shape[2] for start in starts)
        if max(ends) > self.capacity:
            raise ValueError(
                f"KV cache overflow: capacity is {self.capacity:,} tokens."
            )

        for source_row, (slot, start, end) in enumerate(
            zip(rows, starts, ends, strict=True)
        ):
            entry.keys[slot, :, start:end].copy_(keys[source_row])
            entry.values[slot, :, start:end].copy_(values[source_row])

        key_length = max(ends)
        index = torch.tensor(rows, dtype=torch.long, device=self._device)
        return (
            entry.keys.index_select(0, index)[:, :, :key_length],
            entry.values.index_select(0, index)[:, :, :key_length],
        )

    def advance(
        self,
        tokens: int,
        *,
        slots: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        if tokens < 1:
            raise ValueError("Cannot advance the KV cache by fewer than one token.")
        if slots is None:
            end = self.length + tokens
            if end > self.capacity:
                raise ValueError("Cannot advance the KV cache beyond its capacity.")
            self._slot_lengths = [end] * self.batch_size
            self._common_length = end
            return

        rows = self._rows(slots)
        ends = tuple(self._slot_lengths[slot] + tokens for slot in rows)
        if max(ends) > self.capacity:
            raise ValueError("Cannot advance the KV cache beyond its capacity.")
        for slot, end in zip(rows, ends, strict=True):
            self._slot_lengths[slot] = end
        self._common_length = None

    def snapshot(self, length: int | None = None) -> KVCacheSnapshot:
        return self.snapshot_slot(0, length)

    def snapshot_slot(self, slot: int, length: int | None = None) -> KVCacheSnapshot:
        slot = self._slot(slot)
        current_length = self.slot_length(slot)
        length = current_length if length is None else length
        if (
            not isinstance(length, int)
            or isinstance(length, bool)
            or not 0 <= length <= current_length
        ):
            raise ValueError(
                f"Snapshot length must be between 0 and {current_length:,} tokens."
            )
        return KVCacheSnapshot(
            length=length,
            layers=tuple(
                LayerKVSnapshot(
                    keys=layer.keys[slot : slot + 1, :, :length].detach().clone(),
                    values=layer.values[slot : slot + 1, :, :length].detach().clone(),
                )
                for layer in self._layers
            ),
        )

    def snapshot_block(self, start: int, end: int) -> KVBlockSnapshot:
        return self.snapshot_block_slot(0, start, end)

    def snapshot_block_slot(self, slot: int, start: int, end: int) -> KVBlockSnapshot:
        slot = self._slot(slot)
        current_length = self.slot_length(slot)
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= current_length
        ):
            raise ValueError(
                f"Block bounds must satisfy 0 <= start < end <= {current_length:,}."
            )
        return KVBlockSnapshot(
            length=end - start,
            layers=tuple(
                LayerKVSnapshot(
                    keys=layer.keys[slot : slot + 1, :, start:end].detach().clone(),
                    values=layer.values[slot : slot + 1, :, start:end].detach().clone(),
                )
                for layer in self._layers
            ),
        )

    def restore(self, snapshot: KVCacheSnapshot) -> None:
        self.restore_slot(0, snapshot)

    def restore_slot(self, slot: int, snapshot: KVCacheSnapshot) -> None:
        slot = self._slot(slot)
        self._validate_snapshot(snapshot)
        for destination, source in zip(self._layers, snapshot.layers, strict=True):
            destination.keys[slot : slot + 1, :, : snapshot.length].copy_(source.keys)
            destination.values[slot : slot + 1, :, : snapshot.length].copy_(
                source.values
            )
        self._slot_lengths[slot] = snapshot.length
        self._common_length = snapshot.length if self.batch_size == 1 else None

    def restore_blocks(self, blocks: Sequence[KVBlockSnapshot]) -> None:
        self.restore_blocks_into_slot(0, blocks)

    def restore_blocks_into_slot(
        self, slot: int, blocks: Sequence[KVBlockSnapshot]
    ) -> None:
        slot = self._slot(slot)
        total_length = 0
        for block in blocks:
            self._validate_block(block)
            total_length += block.length
            if total_length > self.capacity:
                raise ValueError(
                    f"KV blocks must fit within the {self.capacity:,}-token cache."
                )
        offset = 0
        for block in blocks:
            end = offset + block.length
            for destination, source in zip(self._layers, block.layers, strict=True):
                destination.keys[slot : slot + 1, :, offset:end].copy_(source.keys)
                destination.values[slot : slot + 1, :, offset:end].copy_(source.values)
            offset = end
        self._slot_lengths[slot] = total_length
        self._common_length = total_length if self.batch_size == 1 else None

    def _validate_snapshot(self, snapshot: KVCacheSnapshot) -> None:
        if (
            not isinstance(snapshot.length, int)
            or isinstance(snapshot.length, bool)
            or not 0 <= snapshot.length <= self.capacity
        ):
            raise ValueError(
                f"Snapshot length must fit within the {self.capacity:,}-token cache."
            )
        if len(snapshot.layers) != len(self._layers):
            raise ValueError("Snapshot layer count does not match this Qwen3 model.")
        for destination, source in zip(self._layers, snapshot.layers, strict=True):
            self._validate_layer(destination, source, snapshot.length)

    def _validate_block(self, block: KVBlockSnapshot) -> None:
        if (
            not isinstance(block.length, int)
            or isinstance(block.length, bool)
            or block.length < 1
        ):
            raise ValueError("KV block length must be a positive integer.")
        if len(block.layers) != len(self._layers):
            raise ValueError("KV block layer count does not match this Qwen3 model.")
        for destination, source in zip(self._layers, block.layers, strict=True):
            self._validate_layer(destination, source, block.length)

    @staticmethod
    def _validate_layer(
        destination: LayerKV, source: LayerKVSnapshot, length: int
    ) -> None:
        expected_shape = (
            1,
            destination.keys.shape[1],
            length,
            destination.keys.shape[3],
        )
        if source.keys.shape != expected_shape or source.values.shape != expected_shape:
            raise ValueError("Snapshot tensor shape does not match this KV cache.")
        if (
            source.keys.dtype != destination.keys.dtype
            or source.values.dtype != destination.values.dtype
        ):
            raise ValueError("Snapshot tensor dtype does not match this KV cache.")
        if (
            source.keys.device != destination.keys.device
            or source.values.device != destination.values.device
        ):
            raise ValueError("Snapshot tensor device does not match this KV cache.")

    def _slot(self, slot: int) -> int:
        if (
            not isinstance(slot, int)
            or isinstance(slot, bool)
            or not 0 <= slot < self.batch_size
        ):
            raise ValueError(
                f"KV-cache slot must be between 0 and {self.batch_size - 1}."
            )
        return slot

    def _rows(self, slots: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
        if isinstance(slots, torch.Tensor):
            if slots.ndim != 1:
                raise ValueError(
                    "KV-cache slots must be a non-empty one-dimensional sequence."
                )
            rows = tuple(int(slot) for slot in slots.cpu().tolist())
        else:
            rows = tuple(slots)
        if not rows:
            raise ValueError(
                "KV-cache slots must be a non-empty one-dimensional sequence."
            )
        if len(set(rows)) != len(rows):
            raise ValueError("KV-cache slots must be distinct valid row indexes.")
        for slot in rows:
            self._slot(slot)
        return rows
