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

import logging
import traceback
from typing import Any, Dict

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
import json_numpy
import cv2

from transformers import PreTrainedModel
from .modeling_florence2 import Florence2ForConditionalGeneration
from .transformer import SoftPromptedTransformer
from .action_hub import build_action_space
from .configuration_xvla import XVLAConfig


class XVLA(PreTrainedModel):
    """
    XVLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      • Florence2 encoder-only backbone (vision-language)
      • SoftPromptedTransformer (temporal/action head)
      • Action space (pre/post-processing + loss)
    """
    config_class = XVLAConfig
    base_model_prefix = "xvla"
    supports_gradient_checkpointing = True

    def __init__(self, config: XVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Core settings
        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        # Action space (dimensions + hooks)
        if config.action_mode.lower() == "auto":
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=config.real_action_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)

        # Florence2 backbone (encoder only)
        self.vlm = Florence2ForConditionalGeneration(config.florence_config).to(torch.float32)
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head

        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")

        # Temporal/action head
        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
        )

        # Deferred FastAPI app
        self.app: FastAPI | None = None

    # ============================= Florence2 encoder =============================
    def forward_vlm(
        self,
        input_ids: torch.LongTensor,        # [B, L]
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V] (bool or 0/1)
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text + multi-view images via Florence2 encoder.
        Each sample requires a valid primary view in slot zero.

        Returns:
          { "vlm_features": [B, T_enc, D], "aux_visual_inputs": [B, (V-1)*N, D] }
        """
        B, V = pixel_values.shape[:2]
        if B == 0 or V == 0 or image_mask.shape != (B, V):
            raise ValueError("image_mask must match a non-empty image batch and view axis.")
        if not image_mask[:, 0].to(torch.bool).all():
            raise ValueError("Each sample requires a valid primary image view in slot zero.")
        flat_mask = image_mask.reshape(-1).to(torch.bool)      # [B*V]
        flat_images = pixel_values.flatten(0, 1)                # [B*V, C, H, W]

        valid_images = flat_images[flat_mask]                   # [#valid, C, H, W]
        valid_feats = self.vlm._encode_image(valid_images)      # [#valid, N, D]
        N, D = valid_feats.shape[1:]

        image_features = valid_feats.new_zeros((B * V, N, D))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(B, V, N, D)        # [B, V, N, D]

        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)  # [B, L, D]

        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],  # first view: [B, N, D]
            inputs_embeds,         # [B, L, D]
        )

        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]  # [B, T_enc, D]

        aux_visual_inputs = image_features[:, 1:].reshape(B, -1, D)  # remaining views flattened
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    # ================================= training =================================
    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,  # [B, T=num_actions, D=dim_action]
    ) -> Dict[str, torch.Tensor]:
        """
        1) Encode multimodal inputs.
        2) Diffusion-style noisy mixture of actions: x_t = t*noise + (1-t)*gt.
        3) Space-specific preprocessing, prediction, and supervised loss.
        """
        enc = self.forward_vlm(input_ids, image_input, image_mask)

        B = input_ids.shape[0]
        t = (torch.rand(1, device=input_ids.device)
             + torch.arange(B, device=input_ids.device) / B) % (1 - 1e-5)

        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            **enc,
        )
        return self.action_space.compute_loss(pred_action, action)

    # ================================= inference =================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """
        Iterative denoising (linear schedule).
        Applies action_space.postprocess at the end (e.g., sigmoid on gripper).
        """
        self.eval()
        enc = self.forward_vlm(input_ids, image_input, image_mask)

        B = input_ids.shape[0]
        D = self.action_space.dim_action

        x1 = torch.randn(B, self.num_actions, D, device=proprio.device, dtype=proprio.dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=proprio.device, dtype=proprio.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
        return self.action_space.postprocess(action)

    # =============================== FastAPI service =============================
    def _build_app(self, processor):
        """
        Minimal FastAPI app for XVLA inference.

        Args:
            processor: callable(images, text) -> Dict[str, torch.Tensor]
                       expected keys: "input_ids", "image_input", "image_mask"
        """
        if self.app is not None:
            return
        if not callable(processor):
            raise ValueError("A callable processor is required to start the inference service.")

        app = FastAPI()

        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
                self.eval()
                # Decode consecutive camera slots from the processor's saved layout.
                images = []
                camera_keys = tuple(f"image{index}" for index in range(getattr(processor, "num_views", 3)))
                if any(key.startswith("image") and key[5:].isdigit() and key not in camera_keys for key in payload):
                    return JSONResponse({"error": "Camera slot exceeds the processor's configured view count."}, status_code=400)
                image_keys = [key for key in camera_keys if key in payload]
                if image_keys != list(camera_keys[:len(image_keys)]):
                    return JSONResponse({"error": "Camera slots must be consecutive, starting with image0."}, status_code=400)
                for key in image_keys:
                    v = json_numpy.loads(payload[key])
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1:  # encoded bytes
                            v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                            if v is None:
                                return JSONResponse({"error": f"Could not decode {key}."}, status_code=400)
                            v = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
                        images.append(Image.fromarray(v))
                    elif isinstance(v, (list, tuple)):
                        images.append(Image.fromarray(np.array(v)))
                    elif isinstance(v, str):
                        images.append(Image.open(v))
                    else:
                        return JSONResponse({"error": f"Invalid image payload for {key}."}, status_code=400)
                if not images:
                    return JSONResponse({"error": "No valid images found."}, status_code=400)

                # Multimodal preprocessing by processor
                inputs = processor(images, payload["language_instruction"])
                if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
                    return JSONResponse({"error": "Processor returned incomplete inputs."}, status_code=400)

                # Build proprio/domain tensors
                proprio = torch.tensor(np.asarray(json_numpy.loads(payload["proprio"])))
                domain_value = payload["domain_id"]
                domain = int(domain_value)
                if isinstance(domain_value, float) and domain_value != domain:
                    return JSONResponse({"error": "domain_id must be an integer."}, status_code=400)
                if not 0 <= domain < self.config.num_domains:
                    return JSONResponse({"error": "domain_id is outside the checkpoint's domain range."}, status_code=400)
                domain_id = torch.tensor([domain], dtype=torch.long)
                steps_value = payload.get("steps", 10)
                steps = int(steps_value)
                if isinstance(steps_value, float) and steps_value != steps:
                    return JSONResponse({"error": "steps must be an integer."}, status_code=400)

                # Align to model's device/dtype
                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype
                if proprio.ndim != 1 or proprio.is_complex():
                    return JSONResponse({"error": "proprio must be a real state vector."}, status_code=400)
                proprio = proprio.to(dtype=dtype)
                if not torch.isfinite(proprio).all():
                    return JSONResponse({"error": "proprio must be finite in the model's dtype."}, status_code=400)
                space = getattr(self, "action_space", None)
                if space is not None and space.name in ("ee6d", "joint", "agibot_ee6d", "auto"):
                    width = getattr(space, "dim_proprio", space.dim_action)
                    if proprio.numel() > width or (space.name != "auto" and proprio.numel() != width):
                        return JSONResponse({"error": "proprio width does not match the checkpoint action space."}, status_code=400)

                def to_model(t: torch.Tensor) -> torch.Tensor:
                    if not isinstance(t, torch.Tensor):
                        t = torch.as_tensor(t)
                    # cast floats to model dtype, keep integral/bool as-is
                    return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)

                inputs = {k: to_model(v) for k, v in inputs.items()}
                inputs.update({
                    "proprio": proprio.unsqueeze(0).to(device=device),
                    "domain_id": domain_id.to(device),
                })

                # Inference
                action = self.generate_actions(**inputs, steps=steps).squeeze(0).float().cpu().numpy()
                return JSONResponse({"action": action.tolist()})

            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Request failed"}, status_code=400)

        self.app = app

    def run(self, processor, host: str = "0.0.0.0", port: int = 8000):
        """
        Launch the FastAPI service.
        """
        self._build_app(processor)
        assert self.app is not None
        uvicorn.run(self.app, host=host, port=port)
