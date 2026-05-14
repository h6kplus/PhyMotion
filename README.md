# PhyMotion: Structured 3D Motion Reward for Physics-Grounded Human Video Generation

* Authors: [Yidong Huang](https://owenh-unc.github.io/)\*, [Zun Wang](https://zunwang1.github.io/)\*, [Han Lin](https://hl-hanlin.github.io/), [Dong-Ki Kim](https://dkkim93.github.io/), [Shayegan Omidshafiei](https://www.linkedin.com/in/shayegan/), [Jaehong Yoon](https://jaehong31.github.io/), [Jaemin Cho](https://j-min.io/), [Yue Zhang](https://zhangyuejoslin.github.io/) and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/) (UNC Chapel Hill, FieldAI, NTU Singapore, AI2, Johns Hopkins University)

\* Equal contribution.

* [Project page](https://phy-motion.github.io) · [Arxiv](#) · [Code](https://github.com/h6kplus/PhyMotion)

Generating realistic human motion is a central yet unsolved challenge in video generation. While reinforcement learning (RL)-based post-training has driven recent gains in general video quality, extending it to human motion remains bottlenecked by a reward signal that cannot reliably score motion realism. Existing video rewards primarily rely on 2D perceptual signals, without explicitly modeling the 3D body state, contact, and dynamics underlying articulated human motion, and often assign high scores to videos with floating bodies or physically implausible movements. To address this, we propose PhyMotion, a structured, fine-grained motion reward that grounds recovered 3D human trajectories in a physics simulator and evaluates motion quality along multiple dimensions of physical feasibility. Concretely, we recover SMPL body meshes from generated videos, retarget them onto a humanoid in the MuJoCo physics simulator, and evaluate the resulting motion along three axes: kinematic plausibility, contact and balance consistency, and dynamic feasibility. Each component provides a continuous and interpretable signal tied to a specific aspect of motion quality, allowing the reward to capture which aspects of motion are physically correct or violated. Experiments show that PhyMotion achieves stronger correlation with human judgments than existing reward formulations. These gains carry over to RL-based post-training, where optimizing PhyMotion leads to larger and more consistent improvements than optimizing existing rewards, improving motion realism across both autoregressive and bidirectional video generators under both automatic metrics and blind human evaluation (+68 Elo gain). Ablations show that the three axes provide complementary supervision signals, while the reward preserves overall video generation quality with only modest training overhead.

<p align="center">
<img src="./assets/teaser.jpg" alt="teaser image"/>
</p>


## Environment Setup

1. Create the Python environment and install dependencies.

```
conda create -n phymotion python=3.10
conda activate phymotion
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

2. Install GVHMR. The reward calls GVHMR in-process to recover SMPL-X meshes from generated frames.

```
git clone https://github.com/zju3dv/GVHMR.git /path/to/GVHMR
# Follow GVHMR's README to download inputs/checkpoints/ (~9 GB).
export GVHMR_ROOT=/path/to/GVHMR
```

The training script and the reward module read `GVHMR_ROOT` from the environment.

3. Download the base video generator. We train on top of Causal Forcing 1.3B (the autoregressive distilled version of Wan2.1 T2V-1.3B).

```
mkdir -p checkpoints/casualforcing/chunkwise
# Place causal_forcing.pt here. See the project page for the download link.
```

4. Prepare the training prompt list. We use the MotionX human-motion test split (1123 prompts) for rollout and evaluation. Provide your own one-prompt-per-line file at `dataset/motionx/test.txt`.


## Stage 1: PhyMotion Reward

The reward grounds each generated video in a 3D body and scores it along three feasibility axes (kinematic, contact, dynamic). It is implemented as a single function in `astrolabe/rewards.py`.

| Axis | Sub-scores |
|---|---|
| **Kinematic** | joint velocity, joint acceleration, self-penetration |
| **Contact**   | foot slip, ground penetration, foot float, balance |
| **Dynamic**   | joint torque, ground reaction force, metabolic effort |

The final reward is the mean of the three axes. All feasibility code (joint-based kinematics and MuJoCo-based contact / dynamics) lives in a single file: `astrolabe/scorers/video/smpl_feasibility.py`.

To wire the reward into a config:

```
config.reward_fn = {"phymotion_score": 1.0}
```

To combine with a perceptual reward (e.g. HPSv3) for balanced training:

```
config.reward_fn = {
    "phymotion_score":   1.0,
    "video_hpsv3_local": 1.0,
}
```


## Stage 2: RL Post-Training

Launch RL post-training of Causal Forcing 1.3B with the PhyMotion reward.

```
export GVHMR_ROOT=/path/to/GVHMR
torchrun --nproc_per_node=8 scripts/train_nft_wan.py \
  --config configs/nft_casual_forcing.py:casual_forcing_video_phymotion
```

* `nproc_per_node`: number of GPUs on a single node.

* `--config`: a `<file>:<entry>` selector. The entry `casual_forcing_video_phymotion` uses the PhyMotion reward (see `configs/nft_casual_forcing.py` for other entries that mix in perceptual rewards).

Outputs are written to `logs/nft/<base_model>/<run_name>_<timestamp>/`:

* `checkpoints/checkpoint-<step>/lora/` — PEFT LoRA adapter (rank 256 on `CausalWanAttentionBlock`).

* `optimizer.pt`, `scaler.pt`, and W&B / TensorBoard logs.


## Stage 3: Inference

Roll out a trained LoRA on a list of prompts.

```
torchrun --nproc_per_node=1 scripts/inference_wan.py \
  --base_model checkpoints/casualforcing/chunkwise/causal_forcing.pt \
  --lora_path  logs/nft/wan_casual_chunk/<run_name>/checkpoints/checkpoint-<step> \
  --prompt_file prompts/sample.txt \
  --output_dir outputs/test \
  --num_frames 45 --height 480 --width 832 \
  --guidance_scale 3.0 \
  --denoising_steps "1000,750,500,250" \
  --num_frame_per_block 3 \
  --mixed_precision bf16 --seed 42
```

* `--base_model`: path to the Causal Forcing 1.3B checkpoint.

* `--lora_path`: a `checkpoint-<step>/` folder or its `lora/` subdir.

* `--prompt_file`: a one-prompt-per-line text file.

* `--output_dir`: directory for the generated mp4s.


## Repository Layout

```
phymotion/
├── astrolabe/                       # Reward + scorers
│   ├── rewards.py                   # phymotion_score (headline reward)
│   ├── ema.py
│   ├── stat_tracking.py
│   └── scorers/video/
│       └── smpl_feasibility.py      # SMPL feasibility (kinematic / contact / dynamic)
├── configs/                         # ml_collections training configs
│   ├── base.py
│   ├── _base_clean.py
│   └── nft_casual_forcing.py        # Causal Forcing 1.3B + PhyMotion
├── pipeline/                        # Inference pipelines
│   ├── causal_inference.py          # Block-wise autoregressive sampler w/ KV-cache
│   └── scene_causal_inference.py    # Multi-scene prompt switching
├── scripts/
│   ├── train_nft_wan.py             # RL post-training entry
│   └── inference_wan.py             # T2V rollout with a trained LoRA
├── utils/                           # Wan model wrappers + dataset / loss helpers
├── assets/teaser.jpg
└── requirements.txt
```


## Citation

If you find this work useful, please consider citing:

```bibtex
@article{huang2026phymotion,
  title   = {PhyMotion: Structured 3D Motion Reward for Physics-Grounded Human Video Generation},
  author  = {Huang, Yidong and Wang, Zun and Lin, Han and Kim, Dong-Ki and
             Omidshafiei, Shayegan and Yoon, Jaehong and Cho, Jaemin and
             Zhang, Yue and Bansal, Mohit},
  journal = {arXiv preprint},
  year    = {2026}
}
```


## License

Code is released under the MIT license. The base models, GVHMR, and MotionX prompts retain their own licenses; see their respective repositories for details.
