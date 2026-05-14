"""Bidirectional Wan wrapper for FastWan-style 3-step DMD models.

Uses diffusers' WanTransformer3DModel + UniPCMultistepScheduler internally,
but exposes the same forward(noisy, conditional_dict, timestep) interface as
WanDiffusionWrapper so the rest of the trainer (pipeline, LoRA, etc.) treats
both kinds of generators uniformly.
"""
from typing import List, Optional

import torch
from torch import nn


class BidirectionalWanWrapper(nn.Module):
    """Bidirectional Wan generator backed by diffusers WanTransformer3DModel.

    Designed for FastWan-style 3-step DMD inference:
    - Model output is `pred_noise` (epsilon parameterization at flow time t)
    - Sampling uses UniPCMultistepScheduler with flow_shift=3.0
    - All frames share one timestep (uniform_timestep=True)

    Drop-in replacement for WanDiffusionWrapper(is_causal=False) when the
    underlying weights use the diffusers-style block layout (e.g. FastWan).
    """

    def __init__(
        self,
        model_path: str,
        flow_shift: float = 3.0,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from diffusers import WanTransformer3DModel

        self.model = WanTransformer3DModel.from_pretrained(
            model_path, torch_dtype=torch_dtype
        )
        # Diffusers' from_pretrained(torch_dtype=...) selectively casts: norm
        # params + buffers stay fp32 while linear/conv weights become bf16.
        # For uniform inference dtype, cast everything down explicitly.
        self.model = self.model.to(torch_dtype)
        self.model.eval()
        self.target_dtype = torch_dtype

        self.uniform_timestep = True
        self.flow_shift = flow_shift
        self.seq_len = 32760  # carried over from WanDiffusionWrapper

        # Build a 1000-step sigma table once (matches FlowMatchEulerDiscrete shift).
        # Used by _convert_pred_noise_to_x0 for arbitrary timestep lookups.
        N = 1000
        t = torch.linspace(N, 1, N, dtype=torch.float32)
        t_norm = t / float(N)
        sigmas = flow_shift * t_norm / (1 + (flow_shift - 1) * t_norm)
        self.register_buffer("_table_timesteps", t, persistent=False)
        self.register_buffer("_table_sigmas", sigmas, persistent=False)

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing()

    def get_scheduler(self):
        from diffusers import UniPCMultistepScheduler

        return UniPCMultistepScheduler(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_train_timesteps=1000,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=self.flow_shift,
            final_sigmas_type="zero",
            solver_order=2,
        )

    def _lookup_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Look up sigma_t from the 1000-step table for arbitrary timesteps.
        t: (N,) integer timesteps (1..1000).
        returns: (N, 1, 1, 1) sigmas.
        """
        idx = torch.argmin(
            (self._table_timesteps.to(t.device).unsqueeze(0)
             - t.float().unsqueeze(1)).abs(),
            dim=1,
        )
        return self._table_sigmas.to(t.device)[idx].reshape(-1, 1, 1, 1)

    def _convert_pred_noise_to_x0(
        self,
        pred_noise: torch.Tensor,  # (B*F, C, H, W)
        xt: torch.Tensor,          # (B*F, C, H, W)
        timestep: torch.Tensor,    # (B*F,) or scalar
    ) -> torch.Tensor:
        """Mirror WanDiffusionWrapper._convert_flow_pred_to_x0 but with
        FastWan's flow-shift sigma schedule (not the FlowMatchScheduler one).

        x0 = x_t - sigma_t * pred_noise   (flow matching)
        """
        if timestep.ndim == 0:
            timestep = timestep.expand(pred_noise.shape[0])
        elif timestep.shape[0] == 1:
            timestep = timestep.expand(pred_noise.shape[0])

        sigma_t = self._lookup_sigma(timestep)
        original_dtype = pred_noise.dtype
        x0 = xt.double() - sigma_t.double() * pred_noise.double()
        return x0.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,  # (B, F, C, H, W)
        conditional_dict: dict,
        timestep: torch.Tensor,              # (B, F)
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        scene_cut: bool = False,
    ):
        if kv_cache is not None or crossattn_cache is not None:
            raise NotImplementedError(
                "BidirectionalWanWrapper does not support kv_cache/crossattn_cache; "
                "use WanDiffusionWrapper(is_causal=True) for streaming inference."
            )
        if classify_mode or clean_x is not None:
            raise NotImplementedError(
                "classify_mode / teacher forcing not implemented for bidirectional Wan."
            )

        prompt_embeds = conditional_dict["prompt_embeds"]

        # Diffusers expects (B, C, F, H, W).
        x_bcfhw = noisy_image_or_video.permute(0, 2, 1, 3, 4)

        # All frames share one timestep.
        if self.uniform_timestep:
            t_batch = timestep[:, 0]
        else:
            t_batch = timestep

        pred_noise_bcfhw = self.model(
            hidden_states=x_bcfhw,
            timestep=t_batch,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        pred_noise = pred_noise_bcfhw.permute(0, 2, 1, 3, 4)  # (B, F, C, H, W)

        # Provide pred_x0 for callers that want it (mirrors WanDiffusionWrapper).
        pred_x0 = self._convert_pred_noise_to_x0(
            pred_noise.flatten(0, 1),
            noisy_image_or_video.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, pred_noise.shape[:2])

        return pred_noise, pred_x0
