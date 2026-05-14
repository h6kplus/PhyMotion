from PIL import Image
import os
import sys
import numpy as np
import torch


def aesthetic_score(device):
    from astrolabe.scorers.image.aesthetic import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def video_hpsv2_local(device):
    from astrolabe.scorers.image.hpsv2 import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(videos, prompts, metadata=None):
        # Normalize input to Tensor [B, F, C, H, W] in range [0, 1]
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 1, 4, 2, 3)  # (B,F,H,W,C) -> (B,F,C,H,W)

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0

        batch_size, num_frames, c, h, w = videos.shape

        # Flatten frames and replicate each prompt once per frame
        flat_images = videos.reshape(-1, c, h, w)
        flat_prompts = [p for p in prompts for _ in range(num_frames)]

        # Score in mini-batches to avoid OOM
        reward_batch_size = 8
        all_flat_scores = []
        for i in range(0, len(flat_images), reward_batch_size):
            batch_scores = scorer(flat_images[i:i+reward_batch_size], flat_prompts[i:i+reward_batch_size])
            all_flat_scores.append(batch_scores)
            torch.cuda.empty_cache()

        flat_scores = torch.cat(all_flat_scores, dim=0)
        frame_scores = flat_scores.view(batch_size, num_frames)

        video_rewards = []
        for i in range(batch_size):
            scores = frame_scores[i].tolist()
            # Aggregate via top-30% mean to reduce noise from low-quality frames
            scores.sort(reverse=True)
            top_k = max(1, int(len(scores) * 0.3))
            video_rewards.append(sum(scores[:top_k]) / top_k)

        return np.array(video_rewards, dtype=np.float32), {}

    return _fn


def video_hpsv3_local(device):
    from astrolabe.scorers.video.hpsv3 import HPSv3RewardInferencer

    # HPSv3 is a 7B VLM scorer; it returns (mu, sigma) per sample
    scorer = HPSv3RewardInferencer(
            config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scorers/configs/HPSv3_7B.yaml'),
            checkpoint_path='./reward_ckpts/HPSv3.safetensors',
            device=device
        )

    def _fn(videos, prompts, metadata=None):
        # Normalize input to Tensor [B, F, C, H, W] in range [0, 1]
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 1, 4, 2, 3)  # (B,F,H,W,C) -> (B,F,C,H,W)

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0

        batch_size, num_frames, c, h, w = videos.shape

        # HPSv3 expects a list of PIL images, so convert each frame
        flat_images_tensor = videos.reshape(-1, c, h, w)
        flat_images_np = (flat_images_tensor * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        flat_images_np = flat_images_np.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        flat_images_pil = [Image.fromarray(img) for img in flat_images_np]

        flat_prompts = [p for p in prompts for _ in range(num_frames)]

        # Smaller batch size than HPSv2 due to the 7B model's higher VRAM usage
        reward_batch_size = 4
        all_flat_scores = []

        for i in range(0, len(flat_images_pil), reward_batch_size):
            batch_scores = scorer.reward(flat_images_pil[i:i+reward_batch_size], flat_prompts[i:i+reward_batch_size])
            # scorer.reward returns [batch, 2] as (mu, sigma); take mu
            if batch_scores.ndim == 2:
                batch_scores = batch_scores[:, 0]
            all_flat_scores.append(batch_scores.cpu())
            torch.cuda.empty_cache()

        flat_scores = torch.cat(all_flat_scores, dim=0)
        frame_scores = flat_scores.view(batch_size, num_frames)

        video_rewards_top30 = []
        video_rewards_all = []

        for i in range(batch_size):
            scores = frame_scores[i].tolist()
            avg_all = sum(scores) / len(scores)
            video_rewards_all.append(float(avg_all))
            # Top-30% mean reduces sensitivity to outlier frames
            scores_sorted = sorted(scores, reverse=True)
            top_k = max(1, int(len(scores) * 0.3))
            video_rewards_top30.append(sum(scores_sorted[:top_k]) / top_k)

        # Return top-30% as the primary reward; full-frame avg carried in metadata
        return np.array(video_rewards_top30, dtype=np.float32), {"all_frame_avg": video_rewards_all}

    return _fn


def videoalign_score(device, reward_type="Overall", use_grayscale=False):
    from astrolabe.scorers.video.videoalign import VideoAlignScorer

    # reward_type selects the scoring dimension: "Overall", "VQ" (visual quality),
    # "MQ" (motion quality, grayscale-sensitive), or "TA" (text alignment)
    scorer = VideoAlignScorer(device=device, dtype=torch.bfloat16, reward_type=reward_type, use_grayscale=use_grayscale)

    def _fn(videos, prompts, metadata=None):
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 1, 4, 2, 3)
        
        if videos.dtype != torch.uint8:
            videos = (videos * 255).round().clamp(0, 255).to(torch.uint8)

        scores_tensor = scorer(list(videos), prompts)
        return scores_tensor.tolist(), {}

    return _fn


# Convenience wrappers for individual VideoAlign dimensions
def videoalign_vq_score(device):
    return videoalign_score(device, reward_type="VQ", use_grayscale=False)

def videoalign_mq_score(device):
    return videoalign_score(device, reward_type="MQ", use_grayscale=True)

def videoalign_mq_rgb_score(device):
    return videoalign_score(device, reward_type="MQ", use_grayscale=False)

def videoalign_ta_score(device):
    return videoalign_score(device, reward_type="TA", use_grayscale=False)


def motion_smoothness_score(device):
    from astrolabe.scorers.video.flowscorer import OpticalFlowSmoothnessScorer
    scorer = OpticalFlowSmoothnessScorer(dtype=torch.float32, device=device)

    def _fn(videos, prompts, metadata=None):
        # RAFT optical flow scorer expects (B, C, T, H, W), not (B, T, C, H, W)
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 4, 1, 2, 3)  # (B,F,H,W,C) -> (B,C,F,H,W)
        else:
            if videos.ndim == 5 and videos.shape[2] == 3 and videos.shape[1] != 3:
                videos = videos.permute(0, 2, 1, 3, 4)  # (B,F,C,H,W) -> (B,C,F,H,W)

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0

        scores = scorer(videos, prompts)
        return scores.cpu().tolist(), {}

    return _fn

def dynamic_degree_score(device):
    from astrolabe.scorers.video.dynamic_degree import DynamicDegreeScorer
    ckpt_path = ""  # set to checkpoint path if a pretrained model is available
    scorer = DynamicDegreeScorer(device=device, model_path=ckpt_path)

    def _fn(videos, prompts, metadata=None):
        # Normalize to Tensor [B, C, T, H, W] in range [0, 1]
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 4, 1, 2, 3)  # (B,F,H,W,C) -> (B,C,F,H,W)
        else:
            if videos.ndim == 5 and videos.shape[1] != 3 and videos.shape[2] == 3:
                videos = videos.permute(0, 2, 1, 3, 4)  # (B,F,C,H,W) -> (B,C,F,H,W)

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0

        scores = scorer(videos)
        return scores.cpu().tolist(), {}

    return _fn


def unifiedreward_score_sglang(device):
    # Requires a running sglang server:
    #   python -m sglang.launch_server --model-path CodeGoat24/UnifiedReward-7b-v1.5
    #       --api-key flowgrpo --port 17140 --chat-template chatml-llava
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        # Parse "Final Score: X" (1-5) from model free-text output; default to 0 on failure
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")

    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        # Resize to 512x512 to match the model's expected input resolution
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc / 5.0 for sc in score]  # Normalize 1-5 scale to [0, 1]
        return score, {}

    return _fn


def smpl_physics_score(device):
    """Score videos by SMPL physics coherency via in-process GVHMR + SMPLPhysicsChecker.

    Pipeline: video tensor [B,F,C,H,W] -> full-frame bbox -> ViTPose + HMR2 (batched)
              -> GVHMR transformer -> SMPL params -> SMPLPhysicsChecker.

    Skips YOLO detection (uses full-frame bbox) and file I/O for speed.
    All models are loaded once and kept on the reward device.
    """
    GVHMR_ROOT = os.environ.get("GVHMR_ROOT")
    if not GVHMR_ROOT:
        raise RuntimeError(
            "GVHMR_ROOT env var not set. Install GVHMR (https://github.com/zju3dv/GVHMR) "
            "and run e.g. `export GVHMR_ROOT=/path/to/GVHMR`."
        )
    if not os.path.exists(GVHMR_ROOT):
        raise RuntimeError(f"GVHMR_ROOT path not found: {GVHMR_ROOT}")
    sys.path.insert(0, GVHMR_ROOT)
    import hydra
    from hydra import initialize_config_module, compose
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
    from hmr4d.utils.preproc import Extractor, VitPoseExtractor
    from hmr4d.utils.preproc.vitfeat_extractor import get_batch
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.net_utils import detach_to_cpu
    # All SMPL feasibility code lives in a single module
    from astrolabe.scorers.video.smpl_feasibility import (
        SMPLPhysicsChecker,
        _MJKinSim, smpl_params_to_qpos,
        score_trajectory as _score_traj,
    )

    # --- one-time init: GVHMR preprocessing + model ---
    _orig_cwd = os.getcwd()
    os.chdir(GVHMR_ROOT)

    vitpose_extractor = VitPoseExtractor(tqdm_leave=False)
    feat_extractor = Extractor(tqdm_leave=False)

    # Move sub-models to the correct device (they hardcode .cuda() -> cuda:0).
    # Also monkey-patch .cuda() on the models so internal .cuda() calls in the
    # extract methods send data to the correct device instead of cuda:0.
    vitpose_extractor.pose = vitpose_extractor.pose.to(device)
    feat_extractor.extractor = feat_extractor.extractor.to(device)
    _reward_device = device

    # Patch: make torch.Tensor.cuda() respect the reward device within extract calls
    _orig_tensor_cuda = torch.Tensor.cuda

    def _patched_extract_vitpose(imgs, bbx_xys, img_ds=0.5):
        with torch.cuda.device(_reward_device):
            imgs = imgs.to(_reward_device)
            return vitpose_extractor.extract(imgs, bbx_xys, img_ds=img_ds)

    def _patched_extract_features(imgs, bbx_xys, img_ds=0.5):
        with torch.cuda.device(_reward_device):
            imgs = imgs.to(_reward_device)
            return feat_extractor.extract_video_features(imgs, bbx_xys, img_ds=img_ds)

    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        gvhmr_cfg = compose(config_name="demo", overrides=["static_cam=True"])
    gvhmr_model: DemoPL = hydra.utils.instantiate(gvhmr_cfg.model, _recursive_=False)
    gvhmr_model.load_pretrained_model(gvhmr_cfg.ckpt_path)
    gvhmr_model = gvhmr_model.eval().to(device)

    os.chdir(_orig_cwd)

    checker = SMPLPhysicsChecker(verbose=False)
    mj_sim = _MJKinSim()  # MuJoCo kinematic sim used by contact/dynamic feasibility

    def _score_from_pred(pred, target_fps=30.0):
        """Run SMPLPhysicsChecker on GVHMR prediction dict."""
        try:
            smpl_params = pred["smpl_params_global"]
            num_frames = smpl_params["body_pose"].shape[0]
            if num_frames < 2:
                return {"kinematic": 0.0, "contact": 0.0, "dynamic": 0.0, "contact_mujoco": 0.0, "dynamic_mujoco": 0.0, "phymotion_combined": 0.0, "overall": 0.0}

            body_model = checker.body_model
            if body_model is None:
                return {"kinematic": 0.0, "contact": 0.0, "dynamic": 0.0, "contact_mujoco": 0.0, "dynamic_mujoco": 0.0, "phymotion_combined": 0.0, "overall": 0.0}

            betas = smpl_params["betas"]
            if betas.ndim == 1:
                betas = betas.unsqueeze(0).expand(num_frames, -1)
            elif betas.shape[0] == 1:
                betas = betas.expand(num_frames, -1)

            smplx_out = body_model(
                betas=betas.float(),
                global_orient=smpl_params["global_orient"].float(),
                body_pose=smpl_params["body_pose"].float(),
                transl=smpl_params["transl"].float(),
                left_hand_pose=torch.zeros(num_frames, 45).float(),
                right_hand_pose=torch.zeros(num_frames, 45).float(),
                jaw_pose=torch.zeros(num_frames, 3).float(),
                leye_pose=torch.zeros(num_frames, 3).float(),
                reye_pose=torch.zeros(num_frames, 3).float(),
                expression=torch.zeros(num_frames, 10).float(),
                return_full_pose=True,
            )

            vertices = smplx_out.vertices.detach().cpu().numpy()
            joints = smplx_out.joints.detach().cpu().numpy()

            ROT_Y_TO_Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
            vertices = vertices @ ROT_Y_TO_Z.T
            joints = joints @ ROT_Y_TO_Z.T

            min_z = vertices[0, :, 2].min()
            vertices[:, :, 2] -= min_z
            joints[:, :, 2] -= min_z

            faces = body_model.faces
            if isinstance(faces, torch.Tensor):
                faces = faces.detach().cpu().numpy()

            fps = target_fps
            kinematic_metrics = checker._analyze_kinematics(joints, fps, vertices, faces)
            contact_metrics = checker._analyze_contacts(vertices, joints, fps)
            dynamics_metrics = checker._analyze_dynamics(vertices, joints, contact_metrics, fps)

            kinematic_f = checker._compute_kinematic_feasibility(kinematic_metrics)
            contact_f = checker._compute_contact_feasibility(contact_metrics)
            dynamic_f = checker._compute_dynamic_feasibility(dynamics_metrics)
            overall_f = float(np.mean([kinematic_f, contact_f, dynamic_f]))

            # --- Contact / dynamic via MuJoCo inverse dynamics ---
            try:
                qpos = smpl_params_to_qpos(smpl_params, mj_sim.model)
                mj_out = _score_traj(mj_sim, qpos, fps=fps)
                contact_mujoco = float(mj_out["contact"].score)
                dynamic_mujoco = float(mj_out["dynamic"].score)
            except Exception as e:
                # Fall back to joint-only contact/dynamic if MuJoCo retargeting fails (e.g. unstable IK)
                contact_mujoco = contact_f
                dynamic_mujoco = dynamic_f

            # PhyMotion reward: mean of three feasibility axes
            phymotion_combined = float(np.mean([kinematic_f, contact_mujoco, dynamic_mujoco]))

            return {
                "kinematic":  kinematic_f,
                "contact":    contact_f,
                "dynamic":    dynamic_f,
                "contact_mujoco": contact_mujoco,
                "dynamic_mujoco": dynamic_mujoco,
                "phymotion_combined": phymotion_combined,
                "overall":    overall_f,
            }
        except Exception as e:
            import traceback, os
            print(f"[_score_from_pred ERROR pid={os.getpid()}] {type(e).__name__}: {e}")
            traceback.print_exc()
            return {"kinematic": 0.0, "contact": 0.0, "dynamic": 0.0, "contact_mujoco": 0.0, "dynamic_mujoco": 0.0, "phymotion_combined": 0.0, "overall": 0.0}

    def _fn(videos, prompts, metadata=None):
        import cv2

        # Pin this thread's default CUDA device so bare .cuda() calls inside
        # GVHMR (e.g., gvhmr_pl_demo.py:37) resolve to this rank's GPU.
        # This is executed in a ThreadPoolExecutor worker, so the setting is
        # local to this reward function's call (threads inherit the default
        # device but can change it independently).
        torch.cuda.set_device(_reward_device)

        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos)
        if videos.ndim == 4:
            return [0.0] * videos.shape[0], {}

        videos = videos.cpu().float()
        if videos.min() < -0.1:
            videos = (videos + 1.0) / 2.0
        videos = (videos.clamp(0, 1) * 255).to(torch.uint8)

        batch_size, num_frames, C, H, W = videos.shape
        zero_scores = {"kinematic": 0.0, "contact": 0.0, "dynamic": 0.0,
                       "contact_mujoco": 0.0, "dynamic_mujoco": 0.0, "phymotion_combined": 0.0,
                       "overall": 0.0}
        img_ds = 0.5

        # --- Stage 1: Preprocess all frames from all videos at once ---
        # Downsample, create full-frame bbox, crop to 256x256
        all_frames_np = []
        all_bbx_xys = []
        frame_counts = []
        for b in range(batch_size):
            frames_b = videos[b].permute(0, 2, 3, 1).numpy()  # [F, H, W, C]
            ds_h, ds_w = int(H * img_ds), int(W * img_ds)
            ds_frames = np.stack([cv2.resize(f, (ds_w, ds_h)) for f in frames_b])
            all_frames_np.append(ds_frames)

            # Full-frame bbox at downsampled resolution
            cx, cy = ds_w / 2.0, ds_h / 2.0
            size = max(ds_w, ds_h) * 1.0
            bbx = torch.tensor([[cx, cy, size]], dtype=torch.float32).expand(num_frames, -1)
            all_bbx_xys.append(bbx)
            frame_counts.append(num_frames)

        # Concatenate all frames for batch preprocessing
        cat_frames = np.concatenate(all_frames_np, axis=0)  # [B*F, ds_H, ds_W, C]
        cat_bbx_xys = torch.cat(all_bbx_xys, dim=0)        # [B*F, 3]

        # Crop and resize all frames to 256x256 in one call
        imgs, _ = get_batch(cat_frames, cat_bbx_xys, img_ds=1.0, path_type="np")

        # Full-res bbx_xys for ViTPose postprocessing (keypoint coord mapping)
        cat_bbx_xys_fullres = cat_bbx_xys / img_ds

        # --- Stage 2: Batched ViTPose on all frames ---
        all_kp2d = _patched_extract_vitpose(imgs, cat_bbx_xys_fullres)

        # --- Stage 3: Batched HMR2 on all frames ---
        all_f_imgseq = _patched_extract_features(imgs, cat_bbx_xys_fullres)

        # --- Stage 4: Per-video GVHMR + physics ---
        overall_scores = []
        detail_scores = {"smpl_kinematic": [], "smpl_contact": [], "smpl_dynamic": [],
                         "smpl_contact_mujoco": [], "smpl_dynamic_mujoco": [],
                         "phymotion_combined": []}
        offset = 0

        for b in range(batch_size):
            nf = frame_counts[b]
            kp2d_b = all_kp2d[offset:offset + nf]
            f_imgseq_b = all_f_imgseq[offset:offset + nf]
            bbx_xys_b = cat_bbx_xys_fullres[offset:offset + nf]
            offset += nf

            try:
                R_w2c = torch.eye(3).repeat(nf, 1, 1)
                K_fullimg = estimate_K(W, H).repeat(nf, 1, 1)
                cam_angvel = compute_cam_angvel(R_w2c)

                data = {
                    "length": torch.tensor(nf),
                    "bbx_xys": bbx_xys_b.to(_reward_device),
                    "kp2d": kp2d_b.to(_reward_device),
                    "K_fullimg": K_fullimg.to(_reward_device),
                    "cam_angvel": cam_angvel.to(_reward_device),
                    "f_imgseq": f_imgseq_b.to(_reward_device),
                }

                with torch.no_grad():
                    pred = gvhmr_model.predict(data, static_cam=True)
                pred = detach_to_cpu(pred)
                scores = _score_from_pred(pred)
            except Exception as e:
                import traceback, os
                print(f"[smpl_physics_score ERROR rank_device={_reward_device} pid={os.getpid()}] {type(e).__name__}: {e}")
                traceback.print_exc()
                scores = zero_scores

            overall_scores.append(scores["overall"])
            detail_scores["smpl_kinematic"].append(scores["kinematic"])
            detail_scores["smpl_contact"].append(scores["contact"])
            detail_scores["smpl_dynamic"].append(scores["dynamic"])
            detail_scores["smpl_contact_mujoco"].append(scores["contact_mujoco"])
            detail_scores["smpl_dynamic_mujoco"].append(scores["dynamic_mujoco"])
            detail_scores["phymotion_combined"].append(scores["phymotion_combined"])

        return overall_scores, detail_scores

    return _fn


def smpl_dynamic_score(device):
    """Score videos by SMPL dynamic feasibility only (wraps smpl_physics_score)."""
    inner_fn = smpl_physics_score(device)

    def _fn(videos, prompts, metadata=None):
        overall_scores, detail_scores = inner_fn(videos, prompts, metadata)
        # Return dynamic as primary reward; carry all sub-scores in metadata
        return detail_scores["smpl_dynamic"], detail_scores

    return _fn


def smpl_kinematic_score(device):
    """Score videos by SMPL kinematic feasibility only (wraps smpl_physics_score)."""
    inner_fn = smpl_physics_score(device)

    def _fn(videos, prompts, metadata=None):
        overall_scores, detail_scores = inner_fn(videos, prompts, metadata)
        return detail_scores["smpl_kinematic"], detail_scores

    return _fn


def phymotion_score(device):
    """PhyMotion reward: mean(v1.kinematic, v3.contact, v3.dynamic).

    Kinematic feasibility uses joint-level analysis (velocity,
    acceleration, self-collision); contact and dynamic feasibility use MuJoCo inverse
    dynamics. This is the headline reward used in the paper.
    """
    inner_fn = smpl_physics_score(device)

    def _fn(videos, prompts, metadata=None):
        overall_scores, detail_scores = inner_fn(videos, prompts, metadata)
        return detail_scores["phymotion_combined"], detail_scores

    return _fn


def videophy_pc_score(device):
    """Score videos by VideoPhy-2 Physical Consistency.

    Uses the zero-I/O VideoPhyScorer (no mp4 read/write — works directly on
    in-memory tensors). Output is the VLM's discrete 1-5 rating, normalized
    to [0, 1] for advantage stability.
    """
    from astrolabe.scorers.video.videophy import VideoPhyScorer
    scorer = VideoPhyScorer(device=device, dtype=torch.bfloat16, task="pc")

    def _fn(videos, prompts, metadata=None):
        # Accept (B, F, C, H, W) tensor or numpy/list — convert to uint8 (T,3,H,W) list
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos)
            if videos.shape[-1] == 3:  # (B, F, H, W, 3) -> (B, F, 3, H, W)
                videos = videos.permute(0, 1, 4, 2, 3)
        if videos.dtype != torch.uint8:
            videos = (videos.clamp(0, 1) * 255).round().clamp(0, 255).to(torch.uint8)
        video_list = [v for v in videos]  # each: (F, 3, H, W) uint8

        raw = scorer(video_list)  # (B,) in [0, 5]; 0 means parse failure
        # Normalize 1..5 → 0..1; clip parse failures (0) to 0.
        normalized = ((raw - 1.0) / 4.0).clamp(min=0.0, max=1.0)
        return normalized.tolist(), {"videophy_pc_raw": raw.tolist()}

    return _fn


def constant_score(device):
    """Constant-reward baseline for measuring training overhead with reward
    cost set to ~zero (everything in the trainer except the reward model).
    Returns 0.5 for every video, instantly. No GPU work, no model load.
    """
    def _fn(videos, prompts, metadata=None):
        if hasattr(videos, "shape"):
            n = videos.shape[0]
        else:
            n = len(videos)
        return [0.5] * n, {}
    return _fn


def multi_score(device, score_dict):
    score_functions = {
        "aesthetic": aesthetic_score,
        "unifiedreward": unifiedreward_score_sglang,
        "video_hpsv2": video_hpsv2_local,
        "video_hpsv2_local": video_hpsv2_local,
        "video_hpsv3_local": video_hpsv3_local,
        "motion_smoothness_score": motion_smoothness_score,
        "videoalign_score": lambda dev: videoalign_score(dev, "Overall"),
        "videoalign_vq_score": videoalign_vq_score,
        "videoalign_mq_score": videoalign_mq_score,
        "videoalign_ta_score": videoalign_ta_score,
        "dynamic_degree_score": dynamic_degree_score,
        "smpl_physics_score": smpl_physics_score,
        "smpl_dynamic_score": smpl_dynamic_score,
        "smpl_kinematic_score": smpl_kinematic_score,
        "phymotion_score": phymotion_score,
        "videophy_pc_score": videophy_pc_score,
        "constant_score": constant_score,
    }
    score_fns = {name: score_functions[name](device) for name in score_dict}

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, only_strict=True):
        total_scores = []
        score_details = {}

        # Detect if input is video [B, F, C, H, W]
        is_video = isinstance(images, torch.Tensor) and images.ndim == 5

        for score_name, weight in score_dict.items():
            current_score_name = score_name
            if is_video and score_name == "hpsv2":
                current_score_name = "video_hpsv2"
                if "video_hpsv2" not in score_fns:
                    score_fns["video_hpsv2"] = video_hpsv2_local(device)

            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
                # Unpack sub-scores (e.g. smpl_physics returns smpl_kinematic, etc.)
                if isinstance(rewards, dict):
                    for sub_key, sub_vals in rewards.items():
                        score_details[sub_key] = sub_vals
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn


if __name__ == "__main__":
    pass
