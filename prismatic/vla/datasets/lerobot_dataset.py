"""
lerobot_dataset.py

LeRobot Dataset Wrapper for OpenVLA with support for future-frame prediction supervision.
Wraps lerobot.common.datasets.LeRobotDataset to provide future_pixel_values for auxiliary losses.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK


class LeRobotBatchTransform:
    """Transform LeRobot episodes into OpenVLA format with future frames."""

    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        predict_stop_token: bool = True,
        predict_steps: int = 3,
        frame_stack: int = 1,
    ):
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.predict_steps = predict_steps
        self.frame_stack = frame_stack

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a LeRobot sample to OpenVLA format.

        Expected sample keys (from lerobot):
          - "observation.image": [T, H, W, 3]
          - "action": [T, A]
          - "language_instruction": str (or use default)
          - Any other metadata
        """
        # Extract images and actions
        images = sample.get("observation.image", sample.get("observation", {}).get("image"))  # [T, H, W, 3]
        actions = sample.get("action")  # [T, A]

        if images is None or actions is None:
            raise ValueError("Sample missing 'observation.image' or 'action' keys")

        images = np.array(images) if not isinstance(images, np.ndarray) else images
        actions = np.array(actions) if not isinstance(actions, np.ndarray) else actions

        # Get language instruction (fallback to generic if missing)
        lang = sample.get("language_instruction", "do the task")
        if isinstance(lang, bytes):
            lang = lang.decode()
        lang = lang.lower().strip()

        # Current frame is first frame, current action is first action
        current_img = Image.fromarray(images[0].astype(np.uint8))
        current_action = actions[0]

        # Get future actions (up to predict_steps ahead)
        future_actions = actions[1 : 1 + self.predict_steps]
        future_actions_string = "".join(self.action_tokenizer(future_actions))

        # Current action string
        current_action_string = self.action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        # Build prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(current_img)

        # Mask loss for non-action tokens
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        result = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            actions=torch.tensor(actions[0], dtype=torch.float32),
            dataset_name="lerobot",
        )

        # Add future frames for prediction supervision (T+1..T+N)
        future_frames = []
        for i in range(1, min(1 + self.predict_steps, len(images))):
            fut_img = Image.fromarray(images[i].astype(np.uint8))
            fut_pix = self.image_transform(fut_img)
            future_frames.append(fut_pix)

        if len(future_frames) > 0:
            # Stack: [N, C, H, W]
            result["future_pixel_values"] = torch.stack(future_frames)

        return result


class LeRobotDataset(Dataset):
    """Lightweight wrapper around LeRobot dataset for OpenVLA training."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        batch_transform: Optional[LeRobotBatchTransform] = None,
        predict_steps: int = 3,
    ):
        """
        Initialize LeRobot dataset.

        Args:
            dataset_name: Name of LeRobot dataset (e.g., "aloha_sim_transfer_cube_human_demo")
            split: "train", "val", or "test"
            batch_transform: LeRobotBatchTransform instance
            predict_steps: Number of future steps to include
        """
        try:
            from lerobot.common.datasets import LeRobotDataset as LBDataset
        except ImportError:
            raise ImportError("lerobot not installed. Install via: pip install lerobot")

        self.lerobot_ds = LBDataset(dataset_name, split=split)
        self.batch_transform = batch_transform
        self.predict_steps = predict_steps
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self.lerobot_ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.lerobot_ds[idx]
        if self.batch_transform is not None:
            sample = self.batch_transform(sample)
        return sample
