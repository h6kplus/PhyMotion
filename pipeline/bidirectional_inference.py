"""Bidirectional inference pipeline for FastWan-style 3-step DMD models.

Mirrors `CausalInferencePipeline.inference()` so the trainer treats both
generators uniformly. No KV cache, no block-wise sampling — runs the full
sequence through standard diffusers UniPC sampling.
"""
from typing import List, Optional

import torch
from torch import nn

from utils.bidirectional_wan_wrapper import BidirectionalWanWrapper
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper


class BidirectionalInferencePipeline(nn.Module):
    """3-step UniPC DMD inference for FastWan-style bidirectional Wan.

    Compatible interface with `CausalInferencePipeline`:
        pipeline = BidirectionalInferencePipeline(args, device, generator, text_encoder, vae)
        video, latents = pipeline.inference(noise, text_prompts, return_latents=True)
    """

    def __init__(
        self,
        args,
        device: str,
        generator: Optional[BidirectionalWanWrapper] = None,
        text_encoder: Optional[WanTextEncoder] = None,
        vae: Optional[WanVAEWrapper] = None,
    ):
        super().__init__()

        if generator is None:
            model_path = getattr(args, "model_kwargs", {}).get(
                "model_path", "checkpoints/fastwan/transformer"
            )
            flow_shift = getattr(args, "model_kwargs", {}).get("flow_shift", 3.0)
            self.generator = BidirectionalWanWrapper(
                model_path=model_path, flow_shift=flow_shift,
            )
        else:
            self.generator = generator

        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        self.scheduler = self.generator.get_scheduler()
        self.num_inference_steps = len(args.denoising_step_list)
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        self.timesteps = self.scheduler.timesteps  # diffusers picks the schedule
        self.device = device
        self.args = args

        # For interface compatibility with the causal pipeline.
        self.num_frame_per_block = 1  # not actually used; kept to avoid attr errors
        self.frame_seq_length = 1560

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ):
        """
        Args:
            noise: (B, F, C, H, W) initial latent noise
            text_prompts: list of B prompts
            return_latents: if True, also return clean latents
        Returns:
            video: (B, F_pixel, 3, H_pixel, W_pixel) in [0, 1]
            latents (optional): (B, F, C, H, W) clean latents
        """
        if initial_latent is not None:
            raise NotImplementedError(
                "initial_latent (video extension) not supported in bidirectional pipeline."
            )

        device = noise.device
        # Need (B, C, F, H, W) for diffusers transformer.
        latents = noise.permute(0, 2, 1, 3, 4).contiguous().to(torch.float32)

        with torch.no_grad():
            cond = self.text_encoder(text_prompts=text_prompts)
            prompt_embeds = cond["prompt_embeds"]

        # Re-set timesteps each call (in case scheduler state mutated).
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        target_dtype = self.generator.target_dtype
        prompt_embeds = prompt_embeds.to(target_dtype)

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_batch = t.expand(latents.shape[0])
                noise_pred = self.generator.model(
                    hidden_states=latents.to(target_dtype),
                    timestep=t_batch,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False,
                )[0]

        # latents now (B, C, F, H, W). Decode via VAE wrapper which expects (B, F, C, H, W).
        latents_bfchw = latents.permute(0, 2, 1, 3, 4)

        with torch.no_grad():
            video = self.vae.decode_to_pixel(latents_bfchw.to(target_dtype))
        # Match CausalInferencePipeline: normalize to [0, 1].
        video = (video.clamp(-1, 1) + 1) / 2

        if return_latents:
            return video, latents_bfchw
        return video
