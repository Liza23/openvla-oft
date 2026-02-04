"""
openvla.py

PyTorch Module defining OpenVLA as a lightweight wrapper around a PrismaticVLM; defines custom logic around
discretizing actions with the ActionTokenizer.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import LlamaTokenizerFast

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.vlas.prediction import LatentPredictor

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLA(PrismaticVLM):
    def __init__(
        self,
        *args,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        latent_predictor: Optional[LatentPredictor] = None,
        image_decoder: Optional[object] = None,
        flow_matcher: Optional[object] = None,
        latent_predict_steps: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer
        # Optional latent predictor (Option A)
        self.latent_predictor = latent_predictor
        self.image_decoder = image_decoder
        self.flow_matcher = flow_matcher
        self.latent_predict_steps = latent_predict_steps

    @torch.inference_mode()
    def predict_action(
        self, image: Image, instruction: str, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # If the special empty token ('') does not already appear after the colon (':') token in the prompt
            # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Optionally predict latents/images and fuse into patch embeddings before generation (Options A/B/C)
        inputs_embeds_for_generate = None
        if self.latent_predictor is not None:
            # Run vision backbone -> projector to get projected patch embeddings
            with torch.no_grad():
                if isinstance(pixel_values, dict):
                    patch_features = self.vision_backbone({k: v for k, v in pixel_values.items()})
                else:
                    patch_features = self.vision_backbone(pixel_values)

                # Project to LLM embedding space
                projected_patch_embeddings = self.projector(patch_features)

            # Get input token embeddings
            input_embeddings = self.llm_backbone.embed_input_ids(input_ids)

            # Build simple context: mean-patch + mean-text
            mean_patches = projected_patch_embeddings.mean(dim=1)
            mean_text = input_embeddings.mean(dim=1)
            context = torch.cat([mean_patches, mean_text], dim=-1)

            # Autoregressively predict `latent_predict_steps` latents, update context each step
            predicted_latents = []
            predicted_imgs = []
            for _step in range(self.latent_predict_steps):
                pred_latent, pred_latent_llm = self.latent_predictor(context)
                predicted_latents.append(pred_latent)

                # If decoder present, produce an image from latent
                if self.image_decoder is not None:
                    pred_img = self.image_decoder(pred_latent)
                    predicted_imgs.append(pred_img)

                # Update mean_text with predicted latent projection for next-step prediction
                mean_text = mean_text + pred_latent_llm
                context = torch.cat([mean_patches, mean_text], dim=-1)

            # Fuse predicted latents into projected_patch_embeddings (simple additive fusion)
            # Sum all predicted latent projections and add to patches
            # First, map predicted_latents to llm-space using latent_predictor.to_llm if available
            llm_add = None
            for pl in predicted_latents:
                # map latent -> llm space via predictor's `to_llm` projection
                pl_llm = self.latent_predictor.to_llm(pl)
                if llm_add is None:
                    llm_add = pl_llm
                else:
                    llm_add = llm_add + pl_llm

            if llm_add is not None:
                projected_patch_embeddings = projected_patch_embeddings + llm_add.unsqueeze(1)

            # If images predicted, encode them (detached through vision backbone) and add their projection
            if len(predicted_imgs) > 0:
                # stack along time dim: [B, N, C, H, W]
                imgs_stack = torch.stack(predicted_imgs, dim=1)
                # flatten time into batch for encoding
                B, N, C, H, W = imgs_stack.shape
                imgs_flat = imgs_stack.view(B * N, C, H, W)
                with torch.no_grad():
                    img_patch_feats = self.vision_backbone(imgs_flat)
                    img_proj = self.projector(img_patch_feats)
                # average across predicted frames and reshape back
                img_proj = img_proj.view(B, N, img_proj.shape[1], img_proj.shape[2])
                img_proj_mean = img_proj.mean(dim=1)  # [B, P, D]
                projected_patch_embeddings = projected_patch_embeddings + img_proj_mean

            # Build inputs_embeds matching PrismaticVLM.forward fused ordering
            inputs_embeds_for_generate = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            if inputs_embeds_for_generate is None:
                generated_ids = super(PrismaticVLM, self).generate(
                    input_ids=input_ids,                            # Shape: [1, seq]
                    pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                    max_new_tokens=self.get_action_dim(unnorm_key),
                    **kwargs,
                )
            else:
                generated_ids = super(PrismaticVLM, self).generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    inputs_embeds=inputs_embeds_for_generate,
                    max_new_tokens=self.get_action_dim(unnorm_key),
                    **kwargs,
                )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]
