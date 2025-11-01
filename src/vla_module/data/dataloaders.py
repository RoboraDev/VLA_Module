from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla_module.models.pi0.PI0Config import FeatureType, PI0Config, PolicyFeature


OBS_PREFIX = "observation"
OBS_LANGUAGE = f"{OBS_PREFIX}.language"
OBS_LANGUAGE_TOKENS = f"{OBS_LANGUAGE}.tokens"
OBS_LANGUAGE_ATTENTION_MASK = f"{OBS_LANGUAGE}.attention_mask"

try:
	from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - handled at runtime
	AutoTokenizer = None

LOGGER = logging.getLogger(__name__)


@dataclass
class LeRobotDataLoaderConfig:
	"""Configuration describing how to load a LeRobot dataset for training."""

	repo_id: str = "lerobot/pusht"
	root: str | Path | None = None
	episodes: Sequence[int] | None = None
	horizon: int | None = None
	stride: int = 1
	max_sequences: int | None = None
	batch_size: int = 8
	shuffle: bool = True
	num_workers: int = 4
	pin_memory: bool = True
	persistent_workers: bool = False
	drop_last: bool = False
	tokenizer_name: str = "google/paligemma-3b-pt-224"
	tokenizer_max_length: int | None = None
	download_videos: bool = False


class InstructionTokenizer:
	"""Wrap a Hugging Face tokenizer with graceful degradation when unavailable."""

	def __init__(self, model_name: str, max_length: int):
		self.max_length = max_length
		self.tokenizer = None
		self.pad_token_id = 0
		if AutoTokenizer is None:
			LOGGER.warning(
				"transformers not installed; using zeroed language tokens. "
				"Install transformers to tokenize natural-language task strings."
			)
			return

		try:
			self.tokenizer = AutoTokenizer.from_pretrained(model_name)
			if self.tokenizer.pad_token is None:
				self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token or "<pad>"
				self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
			self.pad_token_id = self.tokenizer.pad_token_id
		except Exception as exc:  # pragma: no cover - network/cache issues
			LOGGER.warning(
				"Failed to load tokenizer '%s' (%s). Falling back to zero tokens.",
				model_name,
				exc,
			)
			self.tokenizer = None

	def encode(self, text: str) -> tuple[torch.LongTensor, torch.BoolTensor]:
		if self.tokenizer is None:
			tokens = torch.full((self.max_length,), self.pad_token_id, dtype=torch.long)
			mask = torch.zeros(self.max_length, dtype=torch.bool)
			return tokens, mask

		encoded = self.tokenizer(
			text,
			max_length=self.max_length,
			padding="max_length",
			truncation=True,
			return_tensors="pt",
		)
		tokens = encoded["input_ids"].squeeze(0).to(torch.long)
		mask = encoded["attention_mask"].squeeze(0).to(torch.bool)
		return tokens, mask


class LeRobotSequenceDataset(Dataset):
	"""Wrap a `LeRobotDataset` to expose fixed-horizon action sequences."""

	def __init__(
		self,
		repo_id: str,
		*,
		root: str | Path | None = None,
		episodes: Sequence[int] | None = None,
		horizon: int = 50,
		stride: int = 1,
		max_sequences: int | None = None,
		download_videos: bool = False,
		image_transforms: Callable | None = None,
	) -> None:
		super().__init__()
		self.horizon = max(1, horizon)
		self.base = LeRobotDataset(
			repo_id,
			root=root,
			episodes=list(episodes) if episodes is not None else None,
			download_videos=download_videos,
			image_transforms=image_transforms,
		)
		self.stride = max(1, stride)
		self.max_sequences = max_sequences
		self.image_keys = list(self.base.meta.camera_keys)
		self.state_key = "observation.state"
		if self.state_key not in self.base.features:
			raise KeyError(f"Dataset '{repo_id}' lacks '{self.state_key}'.")
		self._sequence_index = self._build_index()

	@property
	def meta(self):  # pragma: no cover - passthrough helper
		return self.base.meta

	def _build_index(self) -> list[tuple[int, int, int]]:
		spans: list[tuple[int, int, int]] = []
		starts = self.base.episode_data_index["from"].tolist()
		ends = self.base.episode_data_index["to"].tolist()
		for episode_idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
			if end <= start:
				continue
			cursor = start
			while cursor < end:
				seq_end = min(cursor + self.horizon, end)
				spans.append((episode_idx, cursor, seq_end))
				cursor += self.stride
		if self.max_sequences is not None:
			spans = spans[: self.max_sequences]
		if not spans:
			raise ValueError("No valid sequences could be constructed from the dataset.")
		return spans

	def __len__(self) -> int:
		return len(self._sequence_index)

	def __getitem__(self, index: int) -> dict:
		episode_idx, start_idx, seq_end = self._sequence_index[index]
		base_item = self.base[start_idx]

		actions = self._gather_action_sequence(start_idx, seq_end)
		sample = {
			key: base_item[key].to(torch.float32)
			if torch.is_tensor(base_item[key])
			else base_item[key]
			for key in self.image_keys
		}
		sample[self.state_key] = base_item[self.state_key].to(torch.float32)
		sample["action"] = actions
		sample["episode_index"] = base_item["episode_index"].to(torch.long)
		sample["frame_index"] = base_item["frame_index"].to(torch.long)
		sample["task"] = base_item.get("task", "")
		sample["sequence_start"] = start_idx
		sample["episode_id"] = episode_idx
		return sample

	def _gather_action_sequence(self, start_idx: int, seq_end: int) -> torch.Tensor:
		actions: list[torch.Tensor] = []
		for idx in range(start_idx, seq_end):
			action = self.base.hf_dataset[idx]["action"]
			if not torch.is_tensor(action):
				action = torch.tensor(action, dtype=torch.float32)
			else:
				action = action.to(torch.float32)
			actions.append(action)
		if not actions:
			fallback = self.base.hf_dataset[start_idx]["action"]
			if not torch.is_tensor(fallback):
				fallback = torch.tensor(fallback, dtype=torch.float32)
			actions.append(fallback.to(torch.float32))

		while len(actions) < self.horizon:
			actions.append(actions[-1].clone())

		stacked = torch.stack(actions[: self.horizon], dim=0)
		return stacked


class LeRobotBatchCollator:
	"""Custom collate function preparing model-ready PI0 batches."""

	def __init__(
		self,
		image_keys: Sequence[str],
		state_key: str,
		tokenizer: InstructionTokenizer,
	) -> None:
		self.image_keys = list(image_keys)
		self.state_key = state_key
		self.tokenizer = tokenizer

	def __call__(self, samples: Sequence[dict]) -> dict:
		batch: dict[str, torch.Tensor | list[str]] = {}

		for key in self.image_keys:
			images = torch.stack([self._ensure_channels_first(sample[key]) for sample in samples], dim=0)
			batch[key] = images.to(torch.float32)

		states = torch.stack([sample[self.state_key] for sample in samples], dim=0)
		actions = torch.stack([sample["action"] for sample in samples], dim=0)
		batch[self.state_key] = states.to(torch.float32)
		batch["action"] = actions.to(torch.float32)

		tokens, masks = zip(*(self.tokenizer.encode(sample.get("task", "")) for sample in samples))
		batch[OBS_LANGUAGE_TOKENS] = torch.stack(list(tokens), dim=0)
		batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.stack(list(masks), dim=0)

		batch["episode_index"] = torch.stack([sample["episode_index"] for sample in samples], dim=0)
		batch["frame_index"] = torch.stack([sample["frame_index"] for sample in samples], dim=0)
		batch["tasks"] = [sample.get("task", "") for sample in samples]
		batch["sequence_start"] = torch.tensor([sample["sequence_start"] for sample in samples], dtype=torch.long)
		batch["episode_id"] = torch.tensor([sample["episode_id"] for sample in samples], dtype=torch.long)
		return batch

	@staticmethod
	def _ensure_channels_first(image: torch.Tensor) -> torch.Tensor:
		if image.ndim != 3:
			raise ValueError("Images must have shape [C, H, W] or [H, W, C].")
		if image.shape[0] == 3:
			return image
		if image.shape[-1] == 3:
			return image.permute(2, 0, 1)
		raise ValueError("Unable to infer channel dimension for image tensor.")


def _augment_config_with_dataset(config: PI0Config, dataset: LeRobotSequenceDataset) -> None:
	"""Ensure the PI0 configuration matches dataset features."""

	for key in dataset.image_keys:
		feature = dataset.base.features[key]
		shape = feature.get("shape", ())
		if len(shape) == 3 and shape[-1] == 3:
			c_first = (3, shape[0], shape[1])
		else:
			c_first = tuple(shape)
		config.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=c_first)

	state_dim = dataset.base.features[dataset.state_key]["shape"][0]
	action_dim = dataset.base.features["action"]["shape"][0]

	config.max_state_dim = max(config.max_state_dim, state_dim)
	config.max_action_dim = max(config.max_action_dim, action_dim)

	config.input_features[dataset.state_key] = PolicyFeature(
		type=FeatureType.STATE,
		shape=(state_dim,),
	)
	config.output_features["action"] = PolicyFeature(
		type=FeatureType.ACTION,
		shape=(action_dim,),
	)

	config.validate_features()


def create_lerobot_dataloader(
	config: PI0Config,
	data_cfg: LeRobotDataLoaderConfig,
) -> tuple[LeRobotSequenceDataset, DataLoader]:
	"""Instantiate dataset and dataloader tailored to PI0 training."""

	horizon = data_cfg.horizon or config.chunk_size
	dataset = LeRobotSequenceDataset(
		data_cfg.repo_id,
		root=data_cfg.root,
		episodes=data_cfg.episodes,
		horizon=horizon,
		stride=data_cfg.stride,
		max_sequences=data_cfg.max_sequences,
		download_videos=data_cfg.download_videos,
	)

	_augment_config_with_dataset(config, dataset)

	tokenizer = InstructionTokenizer(
		data_cfg.tokenizer_name,
		max_length=data_cfg.tokenizer_max_length or config.tokenizer_max_length,
	)
	collate_fn = LeRobotBatchCollator(dataset.image_keys, dataset.state_key, tokenizer)

	loader = DataLoader(
		dataset,
		batch_size=data_cfg.batch_size,
		shuffle=data_cfg.shuffle,
		num_workers=data_cfg.num_workers,
		pin_memory=data_cfg.pin_memory,
		persistent_workers=data_cfg.persistent_workers and data_cfg.num_workers > 0,
		drop_last=data_cfg.drop_last,
		collate_fn=collate_fn,
	)
	return dataset, loader


__all__ = [
	"InstructionTokenizer",
	"LeRobotBatchCollator",
	"LeRobotDataLoaderConfig",
	"LeRobotSequenceDataset",
	"create_lerobot_dataloader",
]
