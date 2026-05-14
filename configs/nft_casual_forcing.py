import os
import imp

_base = imp.load_source("_base_clean", os.path.join(os.path.dirname(__file__), "_base_clean.py"))


def get_config(name):
    return globals()[name]()


def _get_config(n_gpus=8, gradient_step_per_epoch=1, dataset="vidprom", reward_fn={}, name=""):
    """Casual Forcing Chunkwise (wan_casual_chunk) 专用配置构建函数"""
    config = _base._make_base_config(dataset)
    config.base_model = "wan_casual_chunk"

    _base._apply_wan_common(config)

    config.pretrained.model = "./checkpoints/casualforcing/chunkwise/causal_forcing.pt"
    config.model_kwargs = {"timestep_shift": 5.0}
    config.sample.num_frame_per_block = 3

    gpu_config = _base._get_gpu_config("wan", n_gpus)
    bsz = gpu_config["bsz"]
    config.sample.num_image_per_prompt = gpu_config["num_image_per_prompt"]
    num_groups = gpu_config["num_groups"]
    config.sample.test_batch_size = gpu_config["test_batch_size"]

    _base._apply_batch_config(config, n_gpus, bsz, num_groups, gradient_step_per_epoch)
    _base._apply_common_fields(config, "wan_casual_chunk", dataset, reward_fn, name)
    return config


# ============================================================================
# 8 GPU
# ============================================================================

def casual_forcing_video_smpl():
    """Casual Forcing + SMPL Physics Coherency - 4 GPU"""
    config = _get_config(
        n_gpus=4, dataset="motionx",
        reward_fn={"smpl_physics_score": 1.0},
        name="casual_forcing_video_smpl",
    )
    config.beta = 0.1
    config.eval_freq = 10
    config.train.learning_rate = 1e-5
    return config


def casual_forcing_video_smpl_dynamic():
    """Casual Forcing + SMPL Dynamic only - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="motionx",
        reward_fn={"smpl_dynamic_score": 1.0},
        name="casual_forcing_video_smpl_dynamic",
    )
    config.beta = 0.1
    config.eval_freq = 10
    config.eval_batches = 5
    config.train.learning_rate = 1e-5
    return config


def casual_forcing_video_phymotion():
    """Causal Forcing 1.3B + PhyMotion reward.

    PhyMotion combines three SMPL feasibility axes: kinematic, contact, and dynamic.
    This is the headline configuration used in the PhyMotion paper.
    """
    config = _get_config(
        n_gpus=8, dataset="motionx",
        reward_fn={"phymotion_score": 1.0},
        name="casual_forcing_video_phymotion",
    )
    config.beta = 0.1
    config.eval_freq = 10
    config.eval_batches = 5
    config.train.learning_rate = 1e-5
    return config


def casual_forcing_video_phymotion_2GPU():
    """Smaller 2-GPU variant of casual_forcing_video_phymotion for quick smoke runs."""
    config = _get_config(
        n_gpus=2, dataset="motionx",
        reward_fn={"phymotion_score": 1.0},
        name="casual_forcing_video_phymotion_2gpu",
    )
    config.beta = 0.1
    config.eval_freq = 10
    config.eval_batches = 5
    config.train.learning_rate = 1e-5
    return config


def casual_forcing_video_smpl_kinematic():
    """Casual Forcing + SMPL Kinematic only - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="motionx",
        reward_fn={"smpl_kinematic_score": 1.0},
        name="casual_forcing_video_smpl_kinematic",
    )
    config.beta = 0.1
    config.eval_freq = 10
    config.eval_batches = 5
    config.train.learning_rate = 1e-5
    return config


def casual_forcing_video_smpl_analysis():
    """Casual Forcing + SMPL Physics - eval only analysis (4 GPU)"""
    config = _get_config(
        n_gpus=4, dataset="motionx",
        reward_fn={"smpl_physics_score": 1.0},
        name="casual_forcing_video_smpl_analysis",
    )
    config.beta = 0.1
    config.eval_freq = 10  # eval every step for analysis
    config.eval_batches = 50  # more batches for thorough analysis
    config.train.learning_rate = 0.0  # no training, just eval
    config.num_epochs = 1
    return config

def casual_forcing_video_hpsv3_2GPU():
    """Casual Forcing + HPSv3 - 8 GPU"""
    config = _get_config(
        n_gpus=2, dataset="motionx",
        reward_fn={"video_hpsv3_local": 1.0},
        name="casual_forcing_video_hpsv3",
    )
    config.beta = 0.1
    config.eval_freq = 10
    config.eval_batches = 5
    config.train.learning_rate = 1e-5
    return config

def casual_forcing_video_hpsv3_4GPU():
    """Casual Forcing + HPSv3 - 8 GPU"""
    config = _get_config(
        n_gpus=4, dataset="motionx",
        reward_fn={"video_hpsv3_local": 1.0},
        name="casual_forcing_video_hpsv3",
    )
    config.beta = 0.1
    config.eval_freq = 10
    config.eval_batches = 5
    config.train.learning_rate = 1e-5
    return config


def casual_forcing_video_hpsv3():
    """Casual Forcing + HPSv3 - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="motionx",
        reward_fn={"video_hpsv3_local": 1.0},
        name="casual_forcing_video_hpsv3",
    )
    config.beta = 0.1
    config.eval_freq = 10
    config.eval_batches = 5
    config.train.learning_rate = 1e-5
    return config


def casual_forcing_video_multi_reward():
    """Casual Forcing + 多奖励 (HPSv3 + MQ) - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 0.1, "videoalign_mq_score": 1.0},
        name="casual_forcing_video_multi_reward",
    )
    config.beta = 0.1
    config.train.beta = 0.0001
    config.eval_freq = 10
    config.train.learning_rate = 2e-5
    return config


def casual_forcing_video_videoalign_mq():
    """Casual Forcing + VideoAlign MQ only (MotionX) - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="motionx",
        reward_fn={"videoalign_mq_score": 1.0},
        name="casual_forcing_video_videoalign_mq",
    )
    config.beta = 0.1
    config.eval_freq = 10
    config.eval_batches = 5
    config.train.learning_rate = 1e-5
    return config


def casual_forcing_video_videophy_pc_2GPU():
    """Casual Forcing + VideoPhy-PC (2 GPU smoke)."""
    config = _get_config(
        n_gpus=2, dataset="motionx",
        reward_fn={"videophy_pc_score": 1.0},
        name="casual_forcing_video_videophy_pc",
    )
    config.beta = 0.1
    config.eval_freq = 10
    config.eval_batches = 5
    config.train.learning_rate = 1e-5
    return config


def casual_forcing_video_videophy_pc():
    """Casual Forcing + VideoPhy-PC only (MotionX) - 8 GPU."""
    config = _get_config(
        n_gpus=8, dataset="motionx",
        reward_fn={"videophy_pc_score": 1.0},
        name="casual_forcing_video_videophy_pc",
    )
    config.beta = 0.1
    config.eval_freq = 10
    config.eval_batches = 5
    config.train.learning_rate = 1e-5
    return config


def casual_forcing_video_videophy_pc_smpl_kin():
    """Casual Forcing + VideoPhy-PC + SMPL Kinematic (MotionX) - 8 GPU.

    Combined reward: SMPL-Kin (continuous, fast, dense gradient) as primary;
    VideoPhy-PC (discrete VLM, semantic physics check) at lower weight to
    catch failure modes SMPL doesn't see (non-human prompts, object physics).
    """
    config = _get_config(
        n_gpus=8, dataset="motionx",
        reward_fn={"smpl_kinematic_score": 1.0, "videophy_pc_score": 0.3},
        name="casual_forcing_video_videophy_pc_smpl_kin",
    )
    config.beta = 0.1
    config.eval_freq = 10
    config.eval_batches = 5
    config.train.learning_rate = 1e-5
    return config


def casual_forcing_video_videoalign_mq_hpsv3():
    """Casual Forcing + VideoAlign MQ + HPSv3 (MotionX) - 8 GPU"""
    config = _get_config(
        n_gpus=8, dataset="motionx",
        reward_fn={"videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="casual_forcing_video_videoalign_mq_hpsv3",
    )
    config.beta = 0.1
    config.eval_freq = 10
    config.eval_batches = 5
    config.train.learning_rate = 1e-5
    return config


# ============================================================================
# 16 GPU
# ============================================================================

def casual_forcing_video_hpsv3_16gpu():
    """Casual Forcing + HPSv3 - 16 GPU"""
    config = _get_config(
        n_gpus=16, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="casual_forcing_video_hpsv3_16gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config


def casual_forcing_video_multi_reward_16gpu():
    """Casual Forcing + 多奖励 (MQ + HPSv3) - 16 GPU"""
    config = _get_config(
        n_gpus=16, dataset="vidprom",
        reward_fn={"videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="casual_forcing_video_multi_reward_16gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config


# ============================================================================
# 24 GPU
# ============================================================================

def casual_forcing_video_hpsv3_24gpu():
    """Casual Forcing + HPSv3 - 24 GPU"""
    config = _get_config(
        n_gpus=24, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="casual_forcing_video_hpsv3_24gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    return config


# ============================================================================
# 48 GPU
# ============================================================================

def casual_forcing_video_hpsv3_48gpu():
    """Casual Forcing + HPSv3 - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0},
        name="casual_forcing_video_hpsv3_48gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


def casual_forcing_video_multireward_48gpu():
    """Casual Forcing + 多奖励 (HPSv3 + MQ) - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"video_hpsv3_local": 1.0, "videoalign_mq_score": 1.0},
        name="casual_forcing_video_multireward_48gpu",
    )
    config.beta = 0.1
    config.train.beta = 0.0005
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


def casual_forcing_video_ta_48gpu():
    """Casual Forcing + Temporal Alignment - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"videoalign_ta_score": 1.0},
        name="casual_forcing_video_ta_48gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 2e-5
    config.eval_freq = 30
    return config


def casual_forcing_video_hpsv3_mq_ta_48gpu():
    """Casual Forcing + HPSv3 + MQ + TA - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"videoalign_ta_score": 1.0, "videoalign_mq_score": 1.0, "video_hpsv3_local": 1.0},
        name="casual_forcing_video_hpsv3_mq_ta_48gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config


def casual_forcing_video_hpsv3_ta_48gpu():
    """Casual Forcing + TA + HPSv3 (0.1 权重) - 48 GPU"""
    config = _get_config(
        n_gpus=48, dataset="vidprom",
        reward_fn={"videoalign_ta_score": 1.0, "video_hpsv3_local": 0.1},
        name="casual_forcing_video_hpsv3_ta_48gpu",
    )
    config.beta = 0.1
    config.train.learning_rate = 1e-5
    config.eval_freq = 30
    return config
