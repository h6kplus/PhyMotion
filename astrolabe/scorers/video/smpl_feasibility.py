"""SMPL feasibility scoring (single-file).

Combines three submodules from the original Astrolabe layout:
  - SMPLPhysicsChecker  (v1 kinematic / contact / dynamic — joint-based)
  - _MJKinSim, smpl_params_to_qpos, helpers (MuJoCo retargeting)
  - score_trajectory     (v3 contact / dynamic via MuJoCo inverse dynamics)

Usage from ``rewards.py``:
    from astrolabe.scorers.video.smpl_feasibility import (
        SMPLPhysicsChecker, _MJKinSim, smpl_params_to_qpos,
        score_trajectory,
    )
"""

from __future__ import annotations
import numpy as np
import os
import scipy.spatial.distance
import smplx
import sys
import torch
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from typing import Dict, List, Optional, Tuple
from typing import Dict, List, Optional, Tuple, Union
from typing import Dict, Optional, Tuple


# ==========================================================================
# from smpl_physics_checker.py
# ==========================================================================

#!/usr/bin/env python3
"""
Direct SMPL Physics Coherency Checker

Analyzes the physical feasibility of SMPL/SMPL-X motion sequences without 
retargeting to robots. Includes kinematic feasibility, contact inference,
balance analysis, and inverse dynamics.
"""


# Human joint ROM data (degrees) - anatomical ranges
HUMAN_ROM_LIMITS = {
    'spine': {
        'flexion_extension': (-20, 45),  # Forward/backward bend
        'lateral_flexion': (-25, 25),    # Side bend
        'rotation': (-30, 30)            # Twist
    },
    'hip': {
        'flexion_extension': (-10, 125),
        'abduction_adduction': (-20, 45),
        'internal_external_rotation': (-45, 45)
    },
    'knee': {
        'flexion_extension': (0, 135),
        'rotation': (-15, 15)
    },
    'ankle': {
        'dorsiflexion_plantarflexion': (-45, 20),
        'inversion_eversion': (-15, 15),
        'rotation': (-20, 20)
    },
    'shoulder': {
        'flexion_extension': (-40, 180),
        'abduction_adduction': (-20, 180),
        'internal_external_rotation': (-90, 90)
    },
    'elbow': {
        'flexion_extension': (0, 145)
    },
    'wrist': {
        'flexion_extension': (-80, 70),
        'radial_ulnar_deviation': (-30, 20)
    },
    'neck': {
        'flexion_extension': (-40, 60),
        'lateral_flexion': (-45, 45),
        'rotation': (-80, 80)
    }
}

# Human segment mass distribution (Dempster-Winter tables)
SEGMENT_MASSES = {
    'head': 0.081,
    'trunk': 0.497,  # torso + neck
    'upper_arm': 0.028,
    'forearm': 0.016,
    'hand': 0.006,
    'thigh': 0.100,
    'shank': 0.0465,
    'foot': 0.0145
}

@dataclass
class KinematicMetrics:
    """Metrics for kinematic feasibility analysis."""
    joint_limit_violations: Dict[str, float]
    velocity_violations: Dict[str, float]
    acceleration_violations: Dict[str, float]
    self_collision_score: float
    total_violation_score: float
    # Soft jitter: continuous acceleration magnitude (lower = smoother)
    jitter_global: float = 0.0
    jitter_local: float = 0.0
    jitter_combined: float = 0.0

@dataclass
class ContactMetrics:
    """Metrics for contact inference and consistency."""
    foot_contact_labels: np.ndarray  # (N, 2) - left/right foot contact
    foot_slip_distances: np.ndarray  # (N, 2) - slip distance per frame
    ground_penetration: np.ndarray   # (N, 2) - penetration depth
    friction_violations: np.ndarray  # (N,) - frames requiring >μ friction
    contact_consistency_score: float
    # Foot floating score (MBench-style): fraction of frames with floating artifacts
    foot_floating_score: float = 0.0

@dataclass  
class DynamicsMetrics:
    """Metrics for inverse dynamics analysis."""
    joint_torques: np.ndarray        # (N, J) - required joint torques
    ground_reaction_forces: np.ndarray  # (N, 6) - GRF [fx,fy,fz,mx,my,mz]
    center_of_mass: np.ndarray       # (N, 3) - COM trajectory
    support_polygon_distances: np.ndarray  # (N,) - COM distance to support
    torque_violations: Dict[str, float]
    grf_violations: Dict[str, float]
    balance_violations: float
    metabolic_cost: float

@dataclass
class PhysicsCoherencyReport:
    """Complete physics coherency analysis report."""
    sequence_length: int
    fps: float
    duration: float
    
    # Feasibility scores (0-1, higher = more feasible)
    kinematic_feasibility: float
    contact_feasibility: float 
    dynamic_feasibility: float
    overall_feasibility: float
    
    # Detailed metrics
    kinematic_metrics: KinematicMetrics
    contact_metrics: ContactMetrics
    dynamics_metrics: DynamicsMetrics
    
    # Summary flags
    is_kinematically_feasible: bool
    is_dynamically_feasible: bool
    is_physically_coherent: bool
    
    def print_summary(self):
        """Print a human-readable summary of the analysis."""
        print(f"\n{'='*60}")
        print("SMPL Physics Coherency Report")
        print(f"{'='*60}")
        print(f"Sequence: {self.sequence_length} frames, {self.duration:.1f}s @ {self.fps:.1f}fps")
        print(f"\nOverall Feasibility: {self.overall_feasibility:.3f}")
        print(f"  Kinematic:  {self.kinematic_feasibility:.3f}")
        print(f"  Contact:    {self.contact_feasibility:.3f}")
        print(f"  Dynamic:    {self.dynamic_feasibility:.3f}")
        
        print(f"\nFeasibility Flags:")
        print(f"  Kinematically feasible: {'✓' if self.is_kinematically_feasible else '✗'}")
        print(f"  Dynamically feasible:   {'✓' if self.is_dynamically_feasible else '✗'}")  
        print(f"  Physically coherent:    {'✓' if self.is_physically_coherent else '✗'}")


class SMPLPhysicsChecker:
    """
    Analyzes physical coherency of SMPL/SMPL-X motion sequences.
    
    This class implements the "apply physics directly to SMPL" approach for
    measuring motion feasibility without robot retargeting.
    """
    
    def __init__(
        self,
        smplx_model_path: str = os.path.join(
            os.environ.get("GVHMR_ROOT", ""),
            "inputs", "checkpoints", "body_models"
        ),
        friction_coefficient: float = 0.7,
        gravity: float = 9.81,
        contact_velocity_threshold: float = 0.05,  # m/s
        contact_height_threshold: float = 0.02,    # m
        verbose: bool = False
    ):
        """
        Initialize SMPL physics checker.
        
        Args:
            smplx_model_path: Path to SMPL-X model files
            friction_coefficient: Ground friction coefficient
            gravity: Gravity acceleration (m/s²)
            contact_velocity_threshold: Max velocity for foot contact (m/s)
            contact_height_threshold: Max height for foot contact (m)
            verbose: Enable verbose output
        """
        self.smplx_model_path = Path(smplx_model_path)
        self.friction_coeff = friction_coefficient
        self.gravity = gravity
        self.contact_vel_thresh = contact_velocity_threshold
        self.contact_height_thresh = contact_height_threshold
        self.verbose = verbose
        
        # Initialize SMPL-X model
        try:
            self.body_model = smplx.create(
                str(self.smplx_model_path), "smplx",
                gender="neutral", use_pca=False
            )
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load SMPL-X model: {e}")
            self.body_model = None

        # Precompute foot sole vertex indices from T-pose
        self.left_foot_indices, self.right_foot_indices = self._find_foot_indices()

    def _find_foot_indices(self):
        """Find foot sole vertex indices from SMPL-X T-pose geometry."""
        if self.body_model is None:
            return [], []
        with torch.no_grad():
            out = self.body_model(return_full_pose=True)
        verts = out.vertices[0].detach().cpu().numpy()
        # Apply Y-up to Z-up rotation (matching analysis pipeline)
        ROT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        verts = verts @ ROT.T
        verts[:, 2] -= verts[:, 2].min()
        # Foot sole: vertices within 4cm of ground
        sole_mask = verts[:, 2] < 0.04
        left_mask = sole_mask & (verts[:, 0] < 0)
        right_mask = sole_mask & (verts[:, 0] > 0)
        return np.where(left_mask)[0].tolist(), np.where(right_mask)[0].tolist()

    def analyze_sequence(
        self, 
        gvhmr_file: str,
        target_fps: float = 30.0,
        target_height: Optional[float] = None
    ) -> PhysicsCoherencyReport:
        """
        Analyze complete physics coherency of a GVHMR motion sequence.
        
        Args:
            gvhmr_file: Path to GVHMR prediction file (.pt)
            target_fps: Target FPS for analysis
            target_height: Optional target human height (m)
            
        Returns:
            Complete physics coherency report
        """
        if self.verbose:
            print(f"\nAnalyzing physics coherency: {gvhmr_file}")
            
        # Load motion data
        vertices, faces, fps, root_positions, joint_positions = self._load_gvhmr_motion(
            gvhmr_file, target_fps, target_height
        )
        
        if self.verbose:
            print(f"Loaded {len(vertices)} frames at {fps:.1f} FPS")
        
        # Analyze kinematics
        kinematic_metrics = self._analyze_kinematics(joint_positions, fps, vertices, faces)
        
        # Analyze contacts  
        contact_metrics = self._analyze_contacts(vertices, joint_positions, fps)
        
        # Analyze dynamics
        dynamics_metrics = self._analyze_dynamics(
            vertices, joint_positions, contact_metrics, fps
        )
        
        # Compute feasibility scores
        kinematic_feasibility = self._compute_kinematic_feasibility(kinematic_metrics)
        contact_feasibility = self._compute_contact_feasibility(contact_metrics)
        dynamic_feasibility = self._compute_dynamic_feasibility(dynamics_metrics)
        
        overall_feasibility = np.mean([
            kinematic_feasibility, contact_feasibility, dynamic_feasibility
        ])
        
        # Determine feasibility flags
        is_kinematically_feasible = kinematic_feasibility > 0.7
        is_dynamically_feasible = dynamic_feasibility > 0.6
        is_physically_coherent = overall_feasibility > 0.65
        
        return PhysicsCoherencyReport(
            sequence_length=len(vertices),
            fps=fps,
            duration=len(vertices) / fps,
            kinematic_feasibility=kinematic_feasibility,
            contact_feasibility=contact_feasibility,
            dynamic_feasibility=dynamic_feasibility,
            overall_feasibility=overall_feasibility,
            kinematic_metrics=kinematic_metrics,
            contact_metrics=contact_metrics,
            dynamics_metrics=dynamics_metrics,
            is_kinematically_feasible=is_kinematically_feasible,
            is_dynamically_feasible=is_dynamically_feasible,
            is_physically_coherent=is_physically_coherent
        )
    
    def get_per_frame_violations(
        self,
        gvhmr_pred: Union[str, dict],
        target_fps: float = 15.0,
    ) -> Dict[str, np.ndarray]:
        """Per-frame boolean violation arrays, derived from the same code the
        scalar score uses. Useful for visualization (highlighting flagged
        frames). All boolean arrays have shape (N,).

        gvhmr_pred can be a path to the cache .pt file or the already-loaded
        cache dict (the same `pred` shape eval_smpl.py uses, i.e. it has
        ['smpl_params_global'] inside).

        Returned keys:
          kin_velocity        max joint vel > 5 m/s
          kin_acceleration    max joint acc > 40 m/s^2
          kin_any             union of kin_*
          dyn_grf_vertical    vertical GRF > 3x body weight
          dyn_grf_horizontal  horizontal GRF > 0.5x body weight
          dyn_balance         COM outside support polygon
          dyn_any             union of dyn_*
        """
        import torch as _torch
        if isinstance(gvhmr_pred, str):
            blob = _torch.load(gvhmr_pred, weights_only=False,
                               map_location="cpu")
            # Caches written by inference + GVHMR are wrapped in {'pred': {...}}
            pred = blob.get("pred", blob)
        else:
            pred = gvhmr_pred
        smpl_params = pred["smpl_params_global"]
        num_frames = smpl_params["body_pose"].shape[0]
        if self.body_model is None:
            raise RuntimeError("SMPL-X body model not loaded")
        betas = smpl_params["betas"]
        if betas.ndim == 1:
            betas = betas.unsqueeze(0).expand(num_frames, -1)
        elif betas.shape[0] == 1:
            betas = betas.expand(num_frames, -1)
        smplx_out = self.body_model(
            betas=betas.float(),
            global_orient=smpl_params["global_orient"].float(),
            body_pose=smpl_params["body_pose"].float(),
            transl=smpl_params["transl"].float(),
            left_hand_pose=_torch.zeros(num_frames, 45).float(),
            right_hand_pose=_torch.zeros(num_frames, 45).float(),
            jaw_pose=_torch.zeros(num_frames, 3).float(),
            leye_pose=_torch.zeros(num_frames, 3).float(),
            reye_pose=_torch.zeros(num_frames, 3).float(),
            expression=_torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )
        vertices = smplx_out.vertices.detach().cpu().numpy()
        joints = smplx_out.joints.detach().cpu().numpy()
        ROT_Y_TO_Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                              dtype=np.float32)
        vertices = vertices @ ROT_Y_TO_Z.T
        joints = joints @ ROT_Y_TO_Z.T
        min_z = vertices[0, :, 2].min()
        vertices[:, :, 2] -= min_z
        joints[:, :, 2] -= min_z

        N = joints.shape[0]
        fps = target_fps
        dt = 1.0 / fps
        # Kinematic per-frame violations (mirrors _analyze_kinematics).
        joint_velocities = np.gradient(joints, dt, axis=0)
        joint_accelerations = np.gradient(joint_velocities, dt, axis=0)
        vel_max_per_frame = np.max(np.linalg.norm(joint_velocities, axis=2),
                                   axis=1)
        acc_max_per_frame = np.max(np.linalg.norm(joint_accelerations, axis=2),
                                   axis=1)
        kin_vel = vel_max_per_frame > 5.0
        kin_acc = acc_max_per_frame > 40.0
        # Contact + Dynamic per-frame (mirrors _analyze_contacts /
        # _analyze_dynamics).
        contact_metrics = self._analyze_contacts(vertices, joints, fps)
        total_mass = 70.0
        com = self._compute_center_of_mass(joints, total_mass)
        com_acc = np.gradient(np.gradient(com, dt, axis=0), dt, axis=0)
        max_vertical_grf = 3.0 * total_mass * self.gravity
        max_horizontal_grf = 0.5 * total_mass * self.gravity
        fz = np.maximum(0.0, total_mass * (self.gravity + com_acc[:, 2]))
        fh = total_mass * np.linalg.norm(com_acc[:, :2], axis=1)
        dyn_grf_v = fz > max_vertical_grf
        dyn_grf_h = fh > max_horizontal_grf
        support_distances = self._analyze_balance(
            com, joints, contact_metrics.foot_contact_labels
        )
        dyn_balance = support_distances > 0.0
        return {
            "kin_velocity":       kin_vel,
            "kin_acceleration":   kin_acc,
            "kin_any":            kin_vel | kin_acc,
            "dyn_grf_vertical":   dyn_grf_v,
            "dyn_grf_horizontal": dyn_grf_h,
            "dyn_balance":        dyn_balance,
            "dyn_any":            dyn_grf_v | dyn_grf_h | dyn_balance,
        }

    def _load_gvhmr_motion(
        self,
        gvhmr_file: str, 
        target_fps: float,
        target_height: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Load GVHMR motion file and extract vertices, joints.
        
        Returns:
            vertices: (N, V, 3) mesh vertices
            faces: (F, 3) triangle faces
            fps: actual FPS
            root_positions: (N, 3) root/pelvis positions
            joint_positions: (N, J, 3) joint positions
        """
        import torch
        from scipy.interpolate import interp1d
        
        # Load GVHMR prediction
        gvhmr_pred = torch.load(gvhmr_file, weights_only=False)
        smpl_params = gvhmr_pred['smpl_params_global']
        
        if self.body_model is None:
            raise RuntimeError("SMPL-X body model not loaded")
        
        # Extract parameters
        num_frames = smpl_params['body_pose'].shape[0]
        betas = smpl_params['betas'][0].numpy()
        
        # Run SMPL-X forward pass
        smplx_output = self.body_model(
            betas=torch.tensor(betas).float().view(1, -1).expand(num_frames, -1),
            global_orient=smpl_params['global_orient'].float(),
            body_pose=smpl_params['body_pose'].float(),
            transl=smpl_params['transl'].float(),
            left_hand_pose=torch.zeros(num_frames, 45).float(),
            right_hand_pose=torch.zeros(num_frames, 45).float(),
            jaw_pose=torch.zeros(num_frames, 3).float(),
            leye_pose=torch.zeros(num_frames, 3).float(),
            reye_pose=torch.zeros(num_frames, 3).float(),
            expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )
        
        # Extract data
        vertices = smplx_output.vertices.detach().cpu().numpy()  # (N, V, 3)
        joints = smplx_output.joints.detach().cpu().numpy()     # (N, J, 3)
        faces = self.body_model.faces
        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()
        
        # Y-up to Z-up conversion (matching GVHMR convention)
        ROT_Y_TO_Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        vertices = vertices @ ROT_Y_TO_Z.T
        joints = joints @ ROT_Y_TO_Z.T
        
        # Ground alignment
        min_z = vertices[0, :, 2].min()
        vertices[:, :, 2] -= min_z
        joints[:, :, 2] -= min_z
        
        # Height scaling
        if target_height is not None:
            current_height = vertices[0, :, 2].max()
            scale = target_height / current_height
            vertices *= scale
            joints *= scale
        
        # FPS alignment via interpolation
        src_fps = 30.0  # GVHMR default
        if abs(target_fps - src_fps) > 0.1:
            frame_ratio = src_fps / target_fps
            new_num_frames = int(num_frames / frame_ratio)
            
            t_orig = np.arange(num_frames)
            t_new = np.linspace(0, num_frames - 1, new_num_frames)
            
            # Interpolate vertices
            vertices_flat = vertices.reshape(num_frames, -1)
            interp_func = interp1d(t_orig, vertices_flat, axis=0, kind='linear')
            vertices = interp_func(t_new).reshape(new_num_frames, -1, 3)
            
            # Interpolate joints
            joints_flat = joints.reshape(num_frames, -1)
            interp_func = interp1d(t_orig, joints_flat, axis=0, kind='linear')
            joints = interp_func(t_new).reshape(new_num_frames, -1, 3)
            
            fps = target_fps
        else:
            fps = src_fps
        
        root_positions = joints[:, 0, :]  # Pelvis is joint 0
        
        return vertices, faces, fps, root_positions, joints
    
    def _analyze_kinematics(self, joint_positions: np.ndarray, fps: float,
                            vertices: np.ndarray = None, faces: np.ndarray = None) -> KinematicMetrics:
        """
        Analyze kinematic feasibility: joint limits, velocities, self-collision, jitter.

        Args:
            joint_positions: (N, J, 3) joint positions
            fps: sequence FPS
            vertices: (N, V, 3) mesh vertices (for BVH self-collision)
            faces: (F, 3) triangle face indices (for BVH self-collision)

        Returns:
            Kinematic analysis metrics
        """
        N, J = joint_positions.shape[:2]

        # Joint velocities and accelerations (finite differences)
        dt = 1.0 / fps
        joint_velocities = np.gradient(joint_positions, dt, axis=0)  # (N, J, 3)
        joint_accelerations = np.gradient(joint_velocities, dt, axis=0)  # (N, J, 3)

        # Joint limit violations (simplified - would need full IK for proper angles)
        joint_limit_violations = self._check_joint_limits(joint_positions)

        # Velocity violations: fraction of frames where ANY joint exceeds threshold
        max_joint_vel = 5.0  # m/s — allows fast sports motion, flags teleportation
        vel_magnitudes = np.linalg.norm(joint_velocities, axis=2)  # (N, J)
        vel_max_per_frame = np.max(vel_magnitudes, axis=1)  # (N,)
        velocity_violations = {'max_joint': float(np.mean(vel_max_per_frame > max_joint_vel))}

        # Acceleration violations: fraction of frames where ANY joint exceeds threshold
        max_joint_acc = 40.0  # m/s²
        acc_magnitudes = np.linalg.norm(joint_accelerations, axis=2)  # (N, J)
        acc_max_per_frame = np.max(acc_magnitudes, axis=1)  # (N,)
        acceleration_violations = {'max_joint': float(np.mean(acc_max_per_frame > max_joint_acc))}

        # Soft jitter: mean joint acceleration magnitude in m/s²
        # Global jitter: acceleration of joints in world space
        global_velocity = joint_positions[1:] - joint_positions[:-1]  # (N-1, J, 3)
        global_acceleration = global_velocity[1:] - global_velocity[:-1]  # (N-2, J, 3)
        global_acceleration = global_acceleration / (dt * dt)  # convert to m/s²
        jitter_global = float(np.linalg.norm(global_acceleration, axis=2).mean()) if N > 2 else 0.0

        # Local jitter: remove root translation, measure joint-relative acceleration
        root = joint_positions[:, 0:1, :]  # (N, 1, 3)
        local_positions = joint_positions - root
        local_velocity = local_positions[1:] - local_positions[:-1]
        local_acceleration = local_velocity[1:] - local_velocity[:-1]
        local_acceleration = local_acceleration / (dt * dt)  # convert to m/s²
        jitter_local = float(np.linalg.norm(local_acceleration, axis=2).mean()) if N > 2 else 0.0

        jitter_combined = jitter_global + jitter_local

        # Self-collision via BVH triangle intersection detection
        if vertices is not None and faces is not None:
            self_collision_score = self._compute_self_collision_score(vertices, faces)
        else:
            self_collision_score = 0.0

        # Total violation score: use soft jitter instead of binary acceleration
        # Jitter baseline ~4.5 m/s² for clean motion; subtract and clamp to get violation
        jitter_baseline = 4.5
        jitter_violation = max(0.0, min(1.0, (jitter_combined - jitter_baseline) / 90.0))

        # Normalize self-collision from percentage (0-100) to 0-1 violation score
        # ~2% is baseline for clean SMPL meshes, 20%+ is severe
        self_collision_violation = max(0.0, min(1.0, (self_collision_score - 2.0) / 18.0))

        total_violations = (
            np.mean(list(joint_limit_violations.values())) +
            np.mean(list(velocity_violations.values())) +
            jitter_violation +
            self_collision_violation
        ) / 4.0

        return KinematicMetrics(
            joint_limit_violations=joint_limit_violations,
            velocity_violations=velocity_violations,
            acceleration_violations=acceleration_violations,
            self_collision_score=self_collision_score,
            total_violation_score=total_violations,
            jitter_global=jitter_global,
            jitter_local=jitter_local,
            jitter_combined=jitter_combined,
        )
    
    def _check_joint_limits(self, joint_positions: np.ndarray) -> Dict[str, float]:
        """Check for anatomically impossible joint configurations."""
        # Simplified joint limit checking using position-based heuristics
        # In practice, this would require full IK to get joint angles
        
        violations = {}
        N = joint_positions.shape[0]
        
        # Example: check if limbs are unnaturally extended/compressed
        # This is a simplified version - real implementation needs joint angles
        
        # Check limb length consistency (sudden changes indicate issues)
        for limb_name, (j1, j2) in [
            ('upper_arm', (16, 18)),  # shoulder to elbow (approximate indices)
            ('forearm', (18, 20)),    # elbow to wrist  
            ('thigh', (1, 4)),        # hip to knee
            ('shank', (4, 7))         # knee to ankle
        ]:
            if j2 < joint_positions.shape[1]:
                limb_lengths = np.linalg.norm(
                    joint_positions[:, j2] - joint_positions[:, j1], axis=1
                )
                length_var = np.std(limb_lengths) / (np.mean(limb_lengths) + 1e-6)
                violations[limb_name] = min(length_var * 10, 1.0)  # Scale to [0,1]
        
        return violations
    
    def _compute_self_collision_score(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """Compute self-collision score using BVH triangle intersection (MBench-style).

        For each frame, builds triangles from vertices + faces, runs BVH collision
        detection, and reports the mean percentage of colliding triangles.

        Args:
            vertices: (N, V, 3) mesh vertices
            faces: (F, 3) triangle face indices

        Returns:
            Mean collision percentage across frames (0-100 scale).
        """
        from mesh_intersection.bvh_search_tree import BVH
        import torch

        N = vertices.shape[0]
        faces_t = torch.as_tensor(faces.astype(np.int64), dtype=torch.long, device='cuda')
        num_triangles = faces_t.shape[0]
        bvh = BVH(max_collisions=8)

        collision_pcts = []
        for i in range(N):
            verts_t = torch.as_tensor(
                vertices[i], dtype=torch.float32, device='cuda'
            ).unsqueeze(0)  # (1, V, 3)
            triangles = verts_t[:, faces_t]  # (1, F, 3, 3)
            outputs = bvh(triangles).detach().cpu().numpy().squeeze()
            collisions = outputs[outputs[:, 0] >= 0, :]
            collision_pcts.append(collisions.shape[0] / float(num_triangles) * 100)

        return float(np.mean(collision_pcts))

    def _compute_foot_floating(
        self,
        joint_positions: np.ndarray,
        foot_positions: np.ndarray,
        foot_contact_labels: np.ndarray,
        fps: float,
    ) -> float:
        """Compute foot floating score using MBench-style analysis.

        Detects three kinds of floating artifacts:
        1. Frame-level: foot moves too fast/slow relative to root during contact
        2. Sequence-level: sustained low relative velocity during non-contact
        3. Mass floating: both feet in air with irregular trajectory

        Args:
            joint_positions: (N, J, 3) joint positions
            foot_positions: (N, 2, 3) foot positions (left, right)
            foot_contact_labels: (N, 2) boolean contact labels
            fps: sequence FPS

        Returns:
            Floating score in [0, 1] — fraction of frames with floating artifacts
        """
        N = joint_positions.shape[0]
        if N < 4:
            return 0.0

        delta_ts = 0.001
        rate_ts = 0.6
        rate_high_ts = 1.75

        # Root position and velocity
        root_pos = joint_positions[:, 0]  # (N, 3)
        root_vel = np.zeros_like(root_pos)
        root_vel[:-1] = root_pos[1:] - root_pos[:-1]
        root_vel[-1] = root_vel[-2]

        # Foot velocity
        foot_vel = np.zeros_like(foot_positions)
        foot_vel[:-1] = foot_positions[1:] - foot_positions[:-1]
        foot_vel[-1] = foot_vel[-2]

        # Relative foot positions and velocities (foot relative to root)
        rel_foot_pos = foot_positions - root_pos[:, np.newaxis, :]
        rel_foot_vel = np.zeros_like(rel_foot_pos)
        rel_foot_vel[:-1] = rel_foot_pos[1:] - rel_foot_pos[:-1]
        rel_foot_vel[-1] = rel_foot_vel[-2]

        # --- 1) Frame-level floating ---
        invalid_flag = np.ones((N, 2))  # 1 = valid, 0 = floating
        left_rates = np.zeros(N)
        right_rates = np.zeros(N)

        for f in range(N):
            root_dis = np.linalg.norm(root_vel[f])
            left_parent_dis = np.linalg.norm(rel_foot_vel[f, 0])
            right_parent_dis = np.linalg.norm(rel_foot_vel[f, 1])
            rate_left = left_parent_dis / (root_dis + 1e-6)
            rate_right = right_parent_dis / (root_dis + 1e-6)
            left_rates[f] = rate_left
            right_rates[f] = rate_right

            left_foot_dis = np.linalg.norm(foot_vel[f, 0])
            right_foot_dis = np.linalg.norm(foot_vel[f, 1])

            if root_dis < delta_ts:
                continue

            lf_l_invalid = rate_left < rate_ts and left_foot_dis > 1.2e-4
            lf_h_invalid = rate_left > rate_high_ts and left_foot_dis > 1.2e-4
            lf_invalid = lf_l_invalid or (lf_h_invalid and root_dis > 1.2e-4)

            rf_l_invalid = rate_right < rate_ts and right_foot_dis > 1.2e-4
            rf_h_invalid = rate_right > rate_high_ts and right_foot_dis > 1.2e-4
            rf_invalid = rf_l_invalid or (rf_h_invalid and root_dis > 1.2e-4)

            contact_sum = int(foot_contact_labels[f, 0]) + int(foot_contact_labels[f, 1])
            if contact_sum == 2 and lf_invalid and rf_invalid:
                invalid_flag[f, 0] = 0
                invalid_flag[f, 1] = 0
            elif foot_contact_labels[f, 0] and not foot_contact_labels[f, 1] and lf_invalid:
                invalid_flag[f, 0] = 0
            elif foot_contact_labels[f, 1] and not foot_contact_labels[f, 0] and rf_invalid:
                invalid_flag[f, 1] = 0

        # --- 2) Sequence-level floating (non-contact ranges) ---
        all_rates = np.stack([left_rates, right_rates], axis=1)  # (N, 2)

        # Get non-contact ranges per foot
        def _get_ranges(contact_col, state):
            """Get contiguous ranges where contact_col == state."""
            ranges = []
            start = -1
            for idx in range(N):
                if bool(contact_col[idx]) == state:
                    if start == -1:
                        start = idx
                    end = idx
                else:
                    if start != -1:
                        ranges.append([start, end])
                        start = -1
            if start != -1:
                ranges.append([start, end])
            return ranges

        no_contact_ranges = [
            _get_ranges(foot_contact_labels[:, 0], False),
            _get_ranges(foot_contact_labels[:, 1], False),
        ]

        floating_range_lens = [0]
        for foot_i in range(2):
            for rge in no_contact_ranges[foot_i]:
                s, e = rge
                rates = all_rates[s:e + 1, foot_i]
                if len(rates) < 4:
                    continue
                # Skip ranges that are mostly stationary
                skip_n = sum(1 for f in range(s, e + 1) if np.linalg.norm(root_vel[f]) < delta_ts)
                if skip_n / (e - s + 1) > 0.5:
                    continue
                # Find contiguous sub-ranges where rate < (rate_ts - 0.2)
                cur_invalid = (rates < (rate_ts - 0.2)).astype(float)
                diff = np.diff(np.concatenate([[0], cur_invalid, [0]]))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                if len(starts) > 0:
                    lengths = ends - starts
                    floating_range_lens.extend(lengths.tolist())

        # --- 3) Mass floating (both feet in air with irregular trajectory) ---
        mass_floating_len = 0
        if no_contact_ranges[0] and no_contact_ranges[1]:
            # Find overlapping non-contact intervals
            for rge0 in no_contact_ranges[0]:
                for rge1 in no_contact_ranges[1]:
                    s = max(rge0[0], rge1[0])
                    e = min(rge0[1], rge1[1])
                    if e - s + 1 < 4:
                        continue
                    # Check if left foot trajectory has > 2 angular peaks
                    start_end_vec = foot_positions[e, 0] - foot_positions[s, 0]
                    start_end_norm = np.linalg.norm(start_end_vec)
                    if start_end_norm < 1e-8:
                        continue
                    agl_list = []
                    for f in range(s + 1, e + 1):
                        cur_vec = foot_positions[f, 0] - foot_positions[s, 0]
                        cos_sim = np.dot(cur_vec, start_end_vec) / (np.linalg.norm(cur_vec) * start_end_norm + 1e-8)
                        cos_sim = np.clip(cos_sim, -1.0, 1.0)
                        agl_list.append(np.degrees(np.abs(np.arccos(cos_sim))))
                    if len(agl_list) >= 3:
                        peaks, _ = find_peaks(agl_list)
                        if len(peaks) > 2:
                            mass_floating_len += e - s + 1

        # --- Combine ---
        merge_invalid = invalid_flag[:, 0] + invalid_flag[:, 1]
        frame_invalid_n = np.sum(merge_invalid <= 1)  # frames where at least one foot is floating

        invalid_n = frame_invalid_n + sum(floating_range_lens) / 2.0 + mass_floating_len
        floating_score = float(invalid_n / N)

        # Bugfix: when both feet have ZERO contact frames, the per-frame logic
        # above can never set invalid_flag=0 (it requires contact_label=True),
        # so the score returns 0.0 (no floating) — falsely reporting clean
        # contact when the body is in fact airborne the whole sequence. Treat
        # this case as "fully floating".
        contact_total = int(foot_contact_labels[:, 0].sum() + foot_contact_labels[:, 1].sum())
        if contact_total == 0:
            return 1.0
        return min(1.0, floating_score)

    def _analyze_contacts(
        self,
        vertices: np.ndarray,
        joint_positions: np.ndarray,
        fps: float
    ) -> ContactMetrics:
        """
        Analyze foot contacts, ground interaction, and foot floating.

        Args:
            vertices: (N, V, 3) mesh vertices
            joint_positions: (N, J, 3) joint positions
            fps: sequence FPS

        Returns:
            Contact analysis metrics
        """
        N = vertices.shape[0]
        dt = 1.0 / fps

        # Use precomputed foot sole vertex indices
        left_foot_indices = self.left_foot_indices
        right_foot_indices = self.right_foot_indices

        # Foot contact detection
        foot_contact_labels = np.zeros((N, 2), dtype=bool)  # [left, right]
        foot_positions = np.zeros((N, 2, 3))

        for i in range(N):
            if len(left_foot_indices) > 0 and max(left_foot_indices) < vertices.shape[1]:
                left_foot_verts = vertices[i, left_foot_indices]
                foot_positions[i, 0] = np.mean(left_foot_verts, axis=0)
                height = np.min(left_foot_verts[:, 2])
                foot_contact_labels[i, 0] = height < self.contact_height_thresh

            if len(right_foot_indices) > 0 and max(right_foot_indices) < vertices.shape[1]:
                right_foot_verts = vertices[i, right_foot_indices]
                foot_positions[i, 1] = np.mean(right_foot_verts, axis=0)
                height = np.min(right_foot_verts[:, 2])
                foot_contact_labels[i, 1] = height < self.contact_height_thresh

        # Refine contact detection with velocity
        foot_velocities = np.gradient(foot_positions, dt, axis=0)  # (N, 2, 3)
        foot_speeds = np.linalg.norm(foot_velocities, axis=2)  # (N, 2)

        for i in range(N):
            for foot in range(2):
                if foot_contact_labels[i, foot] and foot_speeds[i, foot] > self.contact_vel_thresh:
                    foot_contact_labels[i, foot] = False

        # Foot slip analysis
        foot_slip_distances = np.zeros((N, 2))
        for foot in range(2):
            contact_frames = foot_contact_labels[:, foot]
            if np.any(contact_frames):
                slip_velocities = foot_speeds[:, foot].copy()
                slip_velocities[~contact_frames] = 0
                foot_slip_distances[:, foot] = slip_velocities * dt

        # Ground penetration
        ground_penetration = np.zeros((N, 2))
        for i in range(N):
            for foot in range(2):
                foot_indices = left_foot_indices if foot == 0 else right_foot_indices
                if len(foot_indices) > 0 and max(foot_indices) < vertices.shape[1]:
                    min_z = np.min(vertices[i, foot_indices, 2])
                    if min_z < 0:
                        ground_penetration[i, foot] = abs(min_z)

        # Friction violations (simplified)
        friction_violations = np.zeros(N)
        for i in range(N):
            slip_total = np.sum(foot_slip_distances[i])
            contact_total = np.sum(foot_contact_labels[i])
            if contact_total > 0:
                friction_violations[i] = slip_total / (0.01 + contact_total)

        # --- Foot floating (MBench-style) ---
        foot_floating_score = self._compute_foot_floating(
            joint_positions, foot_positions, foot_contact_labels, fps
        )

        # Contact consistency score (now includes foot floating)
        contact_consistency_score = 1.0 - np.mean([
            np.mean(foot_slip_distances),
            np.mean(ground_penetration),
            np.mean(friction_violations > 0.05),
            foot_floating_score,
        ])

        return ContactMetrics(
            foot_contact_labels=foot_contact_labels,
            foot_slip_distances=foot_slip_distances,
            ground_penetration=ground_penetration,
            friction_violations=friction_violations,
            contact_consistency_score=max(0.0, contact_consistency_score),
            foot_floating_score=foot_floating_score,
        )
    
    def _analyze_dynamics(
        self,
        vertices: np.ndarray,
        joint_positions: np.ndarray, 
        contact_metrics: ContactMetrics,
        fps: float
    ) -> DynamicsMetrics:
        """
        Analyze inverse dynamics and force requirements.
        
        Args:
            vertices: (N, V, 3) mesh vertices
            joint_positions: (N, J, 3) joint positions
            contact_metrics: Previously computed contact metrics
            fps: sequence FPS
            
        Returns:
            Dynamics analysis metrics
        """
        N, J = joint_positions.shape[:2]
        dt = 1.0 / fps
        
        # Estimate body mass (typical human ~70kg)
        total_mass = 70.0  # kg
        
        # Compute center of mass (simplified using joint positions)
        # In reality, this needs proper segment masses and CoM locations
        center_of_mass = self._compute_center_of_mass(joint_positions, total_mass)
        
        # COM kinematics
        com_velocity = np.gradient(center_of_mass, dt, axis=0)
        com_acceleration = np.gradient(com_velocity, dt, axis=0)
        
        # Support polygon analysis
        support_distances = self._analyze_balance(
            center_of_mass, joint_positions, contact_metrics.foot_contact_labels
        )
        
        # Simplified inverse dynamics (neglecting joint torques for now)
        # Ground reaction forces from Newton's laws
        ground_reaction_forces = np.zeros((N, 6))  # [fx, fy, fz, mx, my, mz]
        
        for i in range(N):
            # Vertical force (gravity + vertical acceleration)
            fz = total_mass * (self.gravity + com_acceleration[i, 2])
            ground_reaction_forces[i, 2] = max(0, fz)  # Can't pull on ground
            
            # Horizontal forces (horizontal acceleration)
            ground_reaction_forces[i, 0] = total_mass * com_acceleration[i, 0]
            ground_reaction_forces[i, 1] = total_mass * com_acceleration[i, 1]
        
        # Joint torques (simplified - would need full rigid body dynamics)
        joint_torques = self._estimate_joint_torques(joint_positions, fps, total_mass)
        
        # Analyze violations
        torque_violations = self._analyze_torque_violations(joint_torques)
        grf_violations = self._analyze_grf_violations(ground_reaction_forces, total_mass)
        # Soft balance violation: continuous distance instead of binary >10cm.
        # Per-frame distance from COM-projection to support polygon, clipped at
        # 0.5m and normalized to [0, 1]. Using mean instead of binary fraction
        # gives gradient signal over the whole range — the previous metric pinned
        # to 0.0 or 1.0 for ~85% of all videos across baselines (audit section 1).
        # _analyze_balance returns the sentinel 1.0 when both feet have no
        # contact in a frame; we keep these as max-violation frames (matching
        # the old binary behavior at the extreme), but real measured distances
        # in [0, 0.5m] now contribute their true magnitude instead of being
        # binary-thresholded at 0.1m.
        balance_violations = float(np.mean(np.clip(support_distances, 0.0, 0.5) / 0.5))
        
        # Metabolic cost (simplified)
        metabolic_cost = self._estimate_metabolic_cost(joint_torques, joint_positions, fps)
        
        return DynamicsMetrics(
            joint_torques=joint_torques,
            ground_reaction_forces=ground_reaction_forces,
            center_of_mass=center_of_mass,
            support_polygon_distances=support_distances,
            torque_violations=torque_violations,
            grf_violations=grf_violations,
            balance_violations=balance_violations,
            metabolic_cost=metabolic_cost
        )
    
    def _compute_center_of_mass(self, joint_positions: np.ndarray, total_mass: float) -> np.ndarray:
        """Estimate center of mass from joint positions."""
        N, J = joint_positions.shape[:2]
        center_of_mass = np.zeros((N, 3))
        
        # Simplified: weight joints by approximate segment masses
        # Joint 0 (pelvis) gets highest weight
        weights = np.ones(J)
        weights[0] = 3.0  # Pelvis/trunk
        weights /= np.sum(weights)
        
        for i in range(N):
            center_of_mass[i] = np.average(joint_positions[i], axis=0, weights=weights)
        
        return center_of_mass
    
    def _analyze_balance(
        self, 
        center_of_mass: np.ndarray,
        joint_positions: np.ndarray, 
        foot_contacts: np.ndarray
    ) -> np.ndarray:
        """Analyze balance using support polygon and COM projection."""
        N = center_of_mass.shape[0]
        support_distances = np.zeros(N)
        
        # Get approximate foot joint indices
        left_ankle_idx = 7   # Approximate
        right_ankle_idx = 10  # Approximate
        
        for i in range(N):
            # Build support polygon from contacting feet
            support_points = []
            
            if foot_contacts[i, 0]:  # Left foot contact
                if left_ankle_idx < joint_positions.shape[1]:
                    support_points.append(joint_positions[i, left_ankle_idx, :2])
            
            if foot_contacts[i, 1]:  # Right foot contact  
                if right_ankle_idx < joint_positions.shape[1]:
                    support_points.append(joint_positions[i, right_ankle_idx, :2])
            
            if len(support_points) == 0:
                # No support - bad balance
                support_distances[i] = 1.0
            elif len(support_points) == 1:
                # Single support point
                com_proj = center_of_mass[i, :2]
                support_distances[i] = np.linalg.norm(com_proj - support_points[0])
            else:
                # Multiple support points - compute distance to convex hull
                com_proj = center_of_mass[i, :2]
                support_distances[i] = self._point_to_polygon_distance(
                    com_proj, np.array(support_points)
                )
        
        return support_distances
    
    def _point_to_polygon_distance(self, point: np.ndarray, polygon: np.ndarray) -> float:
        """Compute distance from point to polygon (negative if inside)."""
        if len(polygon) < 3:
            # Line segment case
            return np.linalg.norm(point - polygon[0])
        
        try:
            hull = ConvexHull(polygon)
            # Simple approximation - distance to closest edge
            min_dist = float('inf')
            for simplex in hull.simplices:
                edge = polygon[simplex]
                # Distance from point to line segment
                v = edge[1] - edge[0]
                w = point - edge[0]
                c1 = np.dot(w, v)
                if c1 <= 0:
                    d = np.linalg.norm(w)
                else:
                    c2 = np.dot(v, v)
                    if c1 >= c2:
                        d = np.linalg.norm(point - edge[1])
                    else:
                        b = c1 / c2
                        pb = edge[0] + b * v
                        d = np.linalg.norm(point - pb)
                min_dist = min(min_dist, d)
            return min_dist
        except:
            return np.linalg.norm(point - np.mean(polygon, axis=0))
    
    def _estimate_joint_torques(
        self, 
        joint_positions: np.ndarray, 
        fps: float, 
        total_mass: float
    ) -> np.ndarray:
        """Estimate required joint torques (simplified)."""
        N, J = joint_positions.shape[:2]
        dt = 1.0 / fps
        
        # Joint accelerations
        joint_velocities = np.gradient(joint_positions, dt, axis=0)
        joint_accelerations = np.gradient(joint_velocities, dt, axis=0)
        
        # Simplified torque estimation (would need full dynamics model)
        # Approximation: torque ∝ joint acceleration * segment inertia
        torques = np.zeros((N, J))
        
        for j in range(J):
            acc_magnitudes = np.linalg.norm(joint_accelerations[:, j], axis=1)
            # Approximate inertia scaling
            segment_inertia = 1.0  # kg⋅m² (very rough approximation)
            torques[:, j] = acc_magnitudes * segment_inertia
        
        return torques
    
    def _analyze_torque_violations(self, joint_torques: np.ndarray) -> Dict[str, float]:
        """Analyze violations of human joint torque limits."""
        # Human joint torque limits (very approximate)
        torque_limits = {
            'ankle': 200,    # N⋅m
            'knee': 300,
            'hip': 400, 
            'spine': 200,
            'shoulder': 100,
            'elbow': 80,
            'wrist': 30
        }
        
        violations = {}
        N, J = joint_torques.shape
        
        for j in range(J):
            # Rough mapping to joint types (would need proper joint naming)
            if j < 3:
                limit = torque_limits['ankle']
            elif j < 6:
                limit = torque_limits['knee']  
            elif j < 9:
                limit = torque_limits['hip']
            else:
                limit = torque_limits['spine']
            
            exceeded = np.sum(joint_torques[:, j] > limit)
            violations[f'joint_{j}'] = exceeded / N
        
        return violations
    
    def _analyze_grf_violations(
        self, 
        ground_reaction_forces: np.ndarray, 
        total_mass: float
    ) -> Dict[str, float]:
        """Analyze ground reaction force violations."""
        N = ground_reaction_forces.shape[0]
        
        # Typical GRF limits
        max_vertical_grf = 3.0 * total_mass * self.gravity  # 3x body weight
        max_horizontal_grf = 0.5 * total_mass * self.gravity  # 0.5x body weight
        
        violations = {}
        
        # Vertical force violations
        fz = ground_reaction_forces[:, 2]
        vertical_violations = np.sum(fz > max_vertical_grf)
        violations['vertical_grf'] = vertical_violations / N
        
        # Horizontal force violations  
        fh = np.linalg.norm(ground_reaction_forces[:, :2], axis=1)
        horizontal_violations = np.sum(fh > max_horizontal_grf)
        violations['horizontal_grf'] = horizontal_violations / N
        
        return violations
    
    def _estimate_metabolic_cost(
        self, 
        joint_torques: np.ndarray, 
        joint_positions: np.ndarray,
        fps: float
    ) -> float:
        """Estimate metabolic cost of motion (simplified)."""
        dt = 1.0 / fps
        
        # Joint velocities  
        joint_velocities = np.gradient(joint_positions, dt, axis=0)
        joint_speeds = np.linalg.norm(joint_velocities, axis=2)  # (N, J)
        
        # Power = torque * angular_velocity (very simplified)
        power = joint_torques * joint_speeds  # Approximation
        
        # Total metabolic cost (integrate power over time)
        total_power = np.sum(power, axis=1)  # Sum across joints
        metabolic_cost = np.sum(total_power) * dt  # Integrate over time
        
        return metabolic_cost
    
    def _compute_kinematic_feasibility(self, metrics: KinematicMetrics) -> float:
        """Compute overall kinematic feasibility score (0-1)."""
        # Invert violation scores to get feasibility
        feasibility = 1.0 - metrics.total_violation_score
        return max(0.0, min(1.0, feasibility))
    
    def _compute_contact_feasibility(self, metrics: ContactMetrics) -> float:
        """Compute overall contact feasibility score (0-1)."""
        return metrics.contact_consistency_score
    
    def _compute_dynamic_feasibility(self, metrics: DynamicsMetrics) -> float:
        """Compute overall dynamic feasibility score (0-1)."""
        # Weight different violation types
        torque_score = 1.0 - np.mean(list(metrics.torque_violations.values()))
        grf_score = 1.0 - np.mean(list(metrics.grf_violations.values()))
        balance_score = 1.0 - metrics.balance_violations
        
        # Metabolic cost score (normalize to reasonable range)
        metabolic_score = max(0.0, 1.0 - metrics.metabolic_cost / 10000.0)
        
        dynamic_feasibility = np.mean([
            torque_score, grf_score, balance_score, metabolic_score
        ])
        
        return max(0.0, min(1.0, dynamic_feasibility))

# ==========================================================================
# from smpl_feasibility_mujoco.py
# ==========================================================================

"""Physics-grounded feasibility scorers built on MuJoCo.

Replaces v2's Newton's-law GRF and heuristic balance with **engine-resolved
contacts**. The pipeline:

1. Load PHC's SMPL humanoid MJCF once
   (PHC/phc/data/assets/mjcf/smpl_humanoid.xml). 24 bodies, 71.8 kg, 69
   position-controlled actuators, 25 collision capsules + a floor plane.

2. Convert per-frame GVHMR SMPL params -> MJCF qpos (76-d vector):
   pelvis = freejoint(transl, R(global_orient)),
   each subsequent body = its 3 hinge angles split _x/_y/_z by axis,
   computed from the corresponding axis-angle entry in body_pose by
   Euler('xyz', extrinsic) decomposition.

3. Step the engine in mocap-PD mode: each frame becomes a target qpos for
   the actuators; we sub-step at 500 Hz for dt=1/fps to settle contacts.
   Read back per frame:
     - data.cfrc_ext[Foot_body, :3]  -> ground reaction force per foot
     - data.contact[i] (foot vs floor) -> contact points -> support polygon
     - data.qfrc_actuator              -> torques actually applied
     - foot height + contact normal    -> penetration / floating

4. Aggregate into:
     F_kin_mj (jitter, joint-limit, self-collision, pen)
     F_con_mj (slip, pen, float, balance)  -- balance from real polygon
     F_dyn_mj (s_tau, s_grf, s_friction, s_met)

Everything is computed alongside (not in place of) the existing v1/v2 keys.

Inputs from a GVHMR cache .pt:
    pred = torch.load(...)["pred"]
    smpl_params = pred["smpl_params_global"]
        body_pose: (T, 63)  21-joint axis-angle (excluding pelvis)
        global_orient: (T, 3)
        transl: (T, 3)
        betas: (T, 10) or (10,)

Public API (see end of module):
    rescore_one_cache(cache_path, mjcf_path) -> dict[str, float]
"""




# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical SMPL-X body bone order, matching PHC's smpl_humanoid.xml.
# Index 0 is Pelvis (freejoint). The remaining 23 are sequential hinge
# triples (_x, _y, _z) in qpos.
SMPL_BONE_ORDER = [
    "Pelvis",
    "L_Hip", "R_Hip", "Torso",
    "L_Knee", "R_Knee", "Spine",
    "L_Ankle", "R_Ankle", "Chest",
    "L_Toe", "R_Toe", "Neck",
    "L_Thorax", "R_Thorax", "Head",
    "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist",
    "L_Hand", "R_Hand",
]

# Bones to read GRF + contact from. Floor contact normally engages the
# ankle/toe bodies. We track all of them; foot-floating is decided later.
FOOT_BODIES = ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")

# Joint torque limits (N·m) used in s_tau. PHC's actuator gear is set in
# the XML; we look up effective torque via |qfrc_actuator| against these
# physical limits, not the MJCF gear which is for control.
TORQUE_LIMITS = {
    "Hip":      400.0,
    "Knee":     300.0,
    "Ankle":    200.0,
    "Toe":      100.0,
    "Torso":    200.0,
    "Spine":    200.0,
    "Chest":    200.0,
    "Neck":     100.0,
    "Head":     100.0,
    "Thorax":   150.0,
    "Shoulder": 150.0,
    "Elbow":    100.0,
    "Wrist":     50.0,
    "Hand":      30.0,
}

# Friction coefficient: an honest-to-god physics number. We flag a frame
# if the demanded horizontal friction exceeds this fraction of normal.
FRICTION_MU = 0.7

BODY_MASS = 71.79     # actually populated from the model at load time
GRAVITY = 9.81

DEFAULT_MJCF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets", "mjcf", "smpl_humanoid.xml"
)



@dataclass
class FeasibilityBreakdown:
    score: float
    sub: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SMPL -> MJCF qpos conversion
# ---------------------------------------------------------------------------

def _aa_to_xyz_euler(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) -> intrinsic XYZ Euler angles (..., 3).

    PHC's smpl_humanoid.xml decomposes each body's rotation into three
    hinge joints with axes (1,0,0), (0,1,0), (0,0,1) in that order. To
    feed qpos correctly, we apply rotations sequentially as Rx * Ry * Rz
    on the parent frame, which corresponds to *intrinsic* XYZ Euler.

    Implementation: convert axis-angle -> rotation matrix -> Euler. We use
    scipy if available (most accurate), else a pure-numpy fallback.
    """
    from scipy.spatial.transform import Rotation as R
    flat = aa.reshape(-1, 3)
    R_mats = R.from_rotvec(flat).as_matrix()
    # Intrinsic XYZ (i.e. body-frame X then Y then Z) === scipy's "XYZ".
    eulers = R.from_matrix(R_mats).as_euler("XYZ", degrees=False)
    return eulers.reshape(*aa.shape[:-1], 3)


def smpl_params_to_qpos(smpl_params: Dict, model) -> np.ndarray:
    """Convert one trajectory's SMPL params to a (T, model.nq) qpos array.

    The MJCF's qpos layout:
        qpos[0:3]    = pelvis translation (world)
        qpos[3:7]    = pelvis quaternion (wxyz)
        qpos[7:7+3]  = L_Hip xyz hinge angles
        ...
    Joint order in qpos matches model.jnt_qposadr; we trust mujoco's parse.
    """
    import mujoco

    body_pose = np.asarray(smpl_params["body_pose"])      # (T, 63) or (T, 21, 3)
    global_orient = np.asarray(smpl_params["global_orient"])  # (T, 3)
    transl = np.asarray(smpl_params["transl"])            # (T, 3)
    if body_pose.ndim == 2:
        body_pose = body_pose.reshape(-1, 21, 3)
    T = body_pose.shape[0]

    # Y-up -> Z-up rotation (matching the v1/v2 scorers' convention).
    # GVHMR returns Y-up; PHC's MJCF expects Z-up. We rotate the world frame.
    ROT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    transl_z = transl @ ROT.T

    # Pelvis quaternion: combine Y->Z rotation with global_orient axis-angle.
    from scipy.spatial.transform import Rotation as R
    R_orient = R.from_rotvec(global_orient).as_matrix()      # (T, 3, 3)
    R_world = ROT @ R_orient                                 # apply rotation to body
    quat_xyzw = R.from_matrix(R_world).as_quat()             # (T, 4) xyzw
    # mujoco uses (w, x, y, z)
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)

    # 21 child bones × 3 hinge angles via XYZ Euler decomposition.
    eulers = _aa_to_xyz_euler(body_pose)                     # (T, 21, 3)

    # Build the SMPL bone -> child-of-pelvis index in the order used by
    # MJCF's joint declarations. We use the MJCF's own joint name order to
    # be safe across PHC versions.
    qpos_template = np.zeros(model.nq, dtype=np.float64)
    qpos_template[3] = 1.0   # identity quaternion (w=1)

    # Map joint name -> qposadr.
    joint_qposadr = {}
    for j_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        joint_qposadr[name] = model.jnt_qposadr[j_id]

    # Lift floor (z=0) — apply once-per-traj offset so feet start near
    # ground. Compute the minimum mesh foot z at frame 0, after applying
    # the pelvis quaternion + transl: rather than rerun SMPL-X here, use
    # the cached transl.z directly and let the engine settle.
    qpos_all = np.tile(qpos_template, (T, 1))
    qpos_all[:, 0:3] = transl_z
    qpos_all[:, 3:7] = quat_wxyz

    # SMPL's body_pose has 21 joints (canonical SMPL/SMPL-H ordering, no
    # pelvis, no hand wrists' children). PHC's MJCF declares 23 child
    # bones (adds L_Hand and R_Hand as rigid extensions of L_Wrist and
    # R_Wrist). Bones with no SMPL data stay at 0 deg.
    smpl_articulated = SMPL_BONE_ORDER[1:1 + body_pose.shape[1]]
    for child_idx, bone in enumerate(smpl_articulated):
        for axis_idx, axis_char in enumerate(("x", "y", "z")):
            jname = f"{bone}_{axis_char}"
            if jname not in joint_qposadr:
                continue
            adr = joint_qposadr[jname]
            qpos_all[:, adr] = eulers[:, child_idx, axis_idx]

    return qpos_all


# ---------------------------------------------------------------------------
# MuJoCo simulation
# ---------------------------------------------------------------------------

class _MJSimulator:
    """One MuJoCo model + data instance, reused across many trajectories.

    Configures position-controlled tracking: each actuator's ctrl is set
    to the target qpos for that joint, MuJoCo's PD law (kp, kd from the
    XML's <position> default) drives toward it.
    """

    def __init__(self, mjcf_path: str = DEFAULT_MJCF, target_fps: float = 30.0,
                 engine_dt: float = 0.001, armature: float = 0.05):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        # Smaller dt for tracking stability under noisy targets.
        self.model.opt.timestep = engine_dt
        # Armature per dof boosts implicit integrator stability for high-
        # frequency PD control without touching the XML.
        self.model.dof_armature[:] = np.maximum(
            self.model.dof_armature, armature)
        self.data = mujoco.MjData(self.model)
        self.body_mass = float(self.model.body_mass.sum())
        self.target_fps = target_fps

        # Cache body / joint name lookups.
        self.body_id = {}
        for i in range(self.model.nbody):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            self.body_id[n] = i
        self.foot_body_ids = [self.body_id[b] for b in FOOT_BODIES
                              if b in self.body_id]
        self.floor_geom_id = -1
        for i in range(self.model.ngeom):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if n == "floor":
                self.floor_geom_id = i
                break

        # Map actuator -> joint qposadr for ctrl assignment.
        self.act_to_qposadr = []
        for a_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[a_id, 0]
            self.act_to_qposadr.append(self.model.jnt_qposadr[joint_id])

        # Sub-step count so dt_engine ≈ 2 ms (500 Hz) per outer-frame step.
        self.engine_dt = float(self.model.opt.timestep)
        self.frame_dt = 1.0 / target_fps
        self.substeps = max(1, int(round(self.frame_dt / self.engine_dt)))

        # Actuator torque limits indexed by joint name root (e.g. "L_Hip").
        self.act_torque_limit = np.zeros(self.model.nu)
        # Per-actuator PD gains: heavier joints get larger kp.
        kp_table = {
            "Hip": 300.0, "Knee": 300.0, "Ankle": 200.0, "Toe": 50.0,
            "Torso": 250.0, "Spine": 200.0, "Chest": 250.0,
            "Neck": 100.0, "Head": 50.0,
            "Thorax": 200.0, "Shoulder": 200.0,
            "Elbow": 150.0, "Wrist": 80.0, "Hand": 30.0,
        }
        kd_table = {k: 0.1 * v for k, v in kp_table.items()}  # critical-ish damping
        self._kp = np.zeros(self.model.nu)
        self._kd = np.zeros(self.model.nu)
        for a_id in range(self.model.nu):
            j_id = self.model.actuator_trnid[a_id, 0]
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
            root = jname.rsplit("_", 1)[0].split("_", 1)[-1]
            self.act_torque_limit[a_id] = TORQUE_LIMITS.get(root, 100.0)
            self._kp[a_id] = kp_table.get(root, 100.0)
            self._kd[a_id] = kd_table.get(root, 10.0)

    def reset(self):
        self.mj.mj_resetData(self.model, self.data)

    def settle_initial_pose(self, qpos0: np.ndarray, n_settle: int = 200):
        """Hold qpos0 with the per-joint PD law before recording. Settle
        time is in engine steps (not frames); 200 @ 1ms = 0.2 sec wall.
        """
        self.data.qpos[:] = qpos0
        self.data.qvel[:] = 0.0
        self.mj.mj_forward(self.model, self.data)
        # Lift off the floor only if the lowest BODY geom (excluding the
        # floor plane) is below z=0. Use geom_aabb for a tight bottom-z.
        body_geom_ids = [i for i in range(self.model.ngeom)
                          if i != self.floor_geom_id]
        if body_geom_ids:
            zs = self.data.geom_xpos[body_geom_ids, 2]
            radii = self.model.geom_size[body_geom_ids, 0]
            lowest_body_z = float((zs - radii).min())
            if lowest_body_z < 0.0:
                self.data.qpos[2] += -lowest_body_z + 0.005
                self.mj.mj_forward(self.model, self.data)
        target_qpos = qpos0[self.act_to_qposadr]
        for _ in range(n_settle):
            cur_q = np.array([self.data.qpos[a] for a in self.act_to_qposadr])
            cur_qv = np.array([self.data.qvel[self.model.jnt_dofadr[
                self.model.actuator_trnid[i, 0]]] for i in range(self.model.nu)])
            self.data.ctrl[:] = (self._kp * (target_qpos - cur_q)
                                  - self._kd * cur_qv)
            self.mj.mj_step(self.model, self.data)

    def step_to_target(self, qpos_target: np.ndarray,
                       qvel_target: Optional[np.ndarray] = None,
                       kp: Optional[np.ndarray] = None,
                       kd: Optional[np.ndarray] = None):
        """Track (qpos_target, qvel_target) with per-actuator PD + velocity FF.

        kp and kd default to self.kp / self.kd if None. qvel_target is 0
        if None (pure position tracking).
        """
        if kp is None: kp = self._kp
        if kd is None: kd = self._kd
        target_qpos = qpos_target[self.act_to_qposadr]
        if qvel_target is None:
            target_qvel = np.zeros(self.model.nu)
        else:
            target_qvel = np.array([qvel_target[self.model.jnt_dofadr[
                self.model.actuator_trnid[i, 0]]] for i in range(self.model.nu)])
        for _ in range(self.substeps):
            cur_q = np.array([self.data.qpos[a] for a in self.act_to_qposadr])
            cur_qv = np.array([self.data.qvel[self.model.jnt_dofadr[
                self.model.actuator_trnid[i, 0]]] for i in range(self.model.nu)])
            self.data.ctrl[:] = (kp * (target_qpos - cur_q)
                                 + kd * (target_qvel - cur_qv))
            self.mj.mj_step(self.model, self.data)

    # ---------- per-frame measurements ----------

    def grf_per_foot(self) -> Dict[str, np.ndarray]:
        """Return external contact force on each tracked foot body, world
        frame, as 3-vectors. cfrc_ext = applied force from contacts only
        (no inertial / gravity contribution), so this is the GRF.
        """
        out = {}
        for bname in FOOT_BODIES:
            if bname in self.body_id:
                bi = self.body_id[bname]
                # cfrc_ext is (nbody, 6): first 3 = torque, last 3 = force.
                out[bname] = self.data.cfrc_ext[bi, 3:6].copy()
        return out

    def support_polygon(self) -> np.ndarray:
        """Return current contact points (those involving the floor) as
        an (N, 2) array of (x, y) ground-plane positions. N may be 0.
        """
        pts = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == self.floor_geom_id or g2 == self.floor_geom_id:
                pts.append(c.pos[:2].copy())
        return np.array(pts) if pts else np.zeros((0, 2))

    def actuator_torques(self) -> np.ndarray:
        """Per-actuator torque (post-step). qfrc_actuator is (nv,)."""
        # Convert nv-vector to per-actuator. The model has 1 dof per actuator
        # for hinges, so we read at the actuator's joint dof addr.
        out = np.zeros(self.model.nu)
        for a_id in range(self.model.nu):
            j_id = self.model.actuator_trnid[a_id, 0]
            dof_adr = self.model.jnt_dofadr[j_id]
            out[a_id] = self.data.qfrc_actuator[dof_adr]
        return out

    def com_world(self) -> np.ndarray:
        """World-frame center of mass (sum body_mass * xpos / total_mass)."""
        return (self.model.body_mass[:, None] *
                self.data.xpos).sum(axis=0) / self.body_mass


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _point_in_polygon(p: np.ndarray, poly: np.ndarray) -> Tuple[bool, float]:
    """(inside, distance_to_polygon). Empty polygon -> (False, 1.0)."""
    if poly.shape[0] == 0:
        return False, 1.0
    if poly.shape[0] == 1:
        return False, float(np.linalg.norm(p - poly[0]))
    if poly.shape[0] == 2:
        a, b = poly[0], poly[1]
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9),
                          0.0, 1.0))
        return False, float(np.linalg.norm(p - (a + t * ab)))
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(poly)
        verts = poly[hull.vertices]
    except Exception:
        verts = poly

    n = len(verts)
    inside = True; sign = 0
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        s = 1 if cross > 0 else (-1 if cross < 0 else 0)
        if sign == 0:
            sign = s
        elif s != 0 and s != sign:
            inside = False
            break
    if inside:
        return True, 0.0
    # min dist to any edge
    min_d = min(_seg_dist(p, verts[i], verts[(i + 1) % n])
                for i in range(n))
    return False, float(min_d)


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


def simulate_and_score(
    sim: _MJSimulator,
    qpos_traj: np.ndarray,         # (T, nq)
    *,
    fps: float,
    settle_steps: int = 30,
) -> Dict[str, FeasibilityBreakdown]:
    """Run the trajectory through the engine and return the 4 feasibility
    blocks (kin / con / dyn / overall) as FeasibilityBreakdown objects.
    """
    T = qpos_traj.shape[0]
    assert qpos_traj.shape[1] == sim.model.nq

    sim.reset()
    sim.settle_initial_pose(qpos_traj[0], n_settle=settle_steps)

    # Numerical qvel target per frame (joint-space, nv-d). Used as
    # velocity feedforward in step_to_target. We compute it in the actuator
    # qposadr space (which is 1-D per actuator for hinges, so consistent
    # with the qpos finite differences).
    qvel_targets = np.zeros_like(qpos_traj)
    qvel_targets[:-1] = (qpos_traj[1:] - qpos_traj[:-1]) * fps

    # Per-frame logs
    f_grf_total = np.zeros((T, 3))         # combined foot force, world
    contact_count = np.zeros(T, dtype=int)
    com_xy = np.zeros((T, 2))
    com_to_support = np.full(T, 1.0)
    foot_height_min = np.full(T, np.inf)
    foot_speed = np.zeros(T)
    foot_pos_prev = None
    tau_per_act = np.zeros((T, sim.model.nu))

    for t in range(T):
        # qvel target is in qpos units / sec, but step_to_target indexes
        # qvel via jnt_dofadr; for hinge joints qpos and qvel coincide
        # 1-D (jnt_dofadr == jnt_qposadr - 7 for non-pelvis). Build a
        # nv-shaped array and zero out the freejoint slot.
        qvel_t = np.zeros(sim.model.nv)
        for a_id in range(sim.model.nu):
            j_id = sim.model.actuator_trnid[a_id, 0]
            dof_adr = sim.model.jnt_dofadr[j_id]
            q_adr = sim.act_to_qposadr[a_id]
            qvel_t[dof_adr] = qvel_targets[t, q_adr]
        sim.step_to_target(qpos_traj[t], qvel_target=qvel_t)

        grfs = sim.grf_per_foot()
        if grfs:
            f_grf_total[t] = sum(grfs.values())

        poly = sim.support_polygon()
        contact_count[t] = poly.shape[0]
        com = sim.com_world()
        com_xy[t] = com[:2]
        _, d = _point_in_polygon(com_xy[t], poly)
        com_to_support[t] = d

        foot_pos_now = []
        for bid in sim.foot_body_ids:
            foot_pos_now.append(sim.data.xpos[bid].copy())
            foot_height_min[t] = min(foot_height_min[t],
                                     float(sim.data.xpos[bid, 2]))
        if foot_pos_prev is not None:
            speeds = [np.linalg.norm(p1 - p0) * fps
                      for p0, p1 in zip(foot_pos_prev, foot_pos_now)]
            foot_speed[t] = float(np.mean(speeds)) if speeds else 0.0
        foot_pos_prev = foot_pos_now

        tau_per_act[t] = sim.actuator_torques()

    # ---- s_grf: real GRF magnitudes -----------------------------------
    bw = sim.body_mass * GRAVITY
    F_vert = f_grf_total[:, 2]
    F_horiz = np.linalg.norm(f_grf_total[:, :2], axis=-1)
    v_grf_vert = float((F_vert > 3.0 * bw).mean())
    v_grf_horiz = float((F_horiz > 0.5 * bw).mean())
    s_grf = 1.0 - 0.5 * (v_grf_vert + v_grf_horiz)

    # ---- s_friction: required mu < 0.7 fraction -----------------------
    in_contact_frames = (F_vert > 0.05 * bw)
    if in_contact_frames.sum() > 0:
        ratio = F_horiz[in_contact_frames] / np.maximum(
            F_vert[in_contact_frames], 1e-3)
        v_fric = float((ratio > FRICTION_MU).mean())
    else:
        v_fric = 0.0
    s_fric = 1.0 - v_fric

    # ---- s_tau ---------------------------------------------------------
    over = np.abs(tau_per_act) > sim.act_torque_limit[None, :]
    v_tau_per_j = over.mean(axis=0)
    s_tau = 1.0 - float(v_tau_per_j.mean())

    # ---- s_balance: COM in support polygon ----------------------------
    v_bal = float(np.mean(np.clip(com_to_support, 0.0, 0.5) / 0.5))
    s_bal = 1.0 - v_bal

    # ---- s_pen: foot below floor (engine usually prevents this; signal
    # is residual penetration energy) ----------------------------------
    pen = np.clip(-foot_height_min, 0.0, 0.05)
    v_pen = float(np.mean(pen) / 0.05)

    # ---- s_slip: foot motion while in contact -------------------------
    in_contact_floor = contact_count > 0
    slip = foot_speed * in_contact_floor.astype(float)
    v_slip = float(np.mean(np.clip(slip, 0.0, 0.5) / 0.5))

    # ---- s_float: contacts vanish for extended periods -----------------
    no_contact = (contact_count == 0).astype(float)
    v_float = float(no_contact.mean())

    # NOTE: We do NOT emit a kinematic feasibility from MuJoCo.
    # Kinematic feasibility per the appendix is about angular-velocity
    # plausibility, self-collision, and joint-limit violations — all
    # measured from the raw recovered joints/mesh, not from the
    # simulator (which actually *prevents* these artifacts via PD
    # smoothing and capsule-collision resolution). Callers should pair
    # MuJoCo's contact/dynamic with v1 or v2 kinematic.
    F_con = FeasibilityBreakdown(
        score=float(np.clip(1.0 - 0.25 * (v_slip + v_pen + v_float + v_bal),
                            0.0, 1.0)),
        sub={"v_slip_mj": v_slip, "v_pen_mj": v_pen,
             "v_float_mj": v_float, "v_bal_mj": v_bal,
             "in_contact_frac": float(in_contact_floor.mean()),
             "foot_min_h_mean": float(foot_height_min.mean())},
    )
    F_dyn = FeasibilityBreakdown(
        score=float(np.clip((s_tau + s_grf + s_fric + s_bal) / 4.0,
                            0.0, 1.0)),
        sub={"s_tau_mj": s_tau, "s_grf_mj": s_grf,
             "s_fric_mj": s_fric, "s_bal_mj": s_bal,
             "v_grf_vert_mj": v_grf_vert, "v_grf_horiz_mj": v_grf_horiz,
             "v_fric_mj": v_fric},
    )
    return {
        "contact": F_con,
        "dynamic": F_dyn,
    }


# ---------------------------------------------------------------------------
# Public driver-level helper
# ---------------------------------------------------------------------------

def rescore_one_cache(
    cache_path: str,
    sim: Optional[_MJSimulator] = None,
    *,
    target_fps: float = 30.0,
) -> Optional[Dict[str, float]]:
    """Load a GVHMR cache .pt, build qpos, simulate, return scoring dict.

    Returns None on load failure so callers can skip without aborting.
    """
    import torch as _t
    try:
        blob = _t.load(cache_path, weights_only=False, map_location="cpu")
        pred = blob.get("pred", blob)
        smpl_params = pred["smpl_params_global"]
        if smpl_params["body_pose"].shape[0] < 2:
            return None
        if sim is None:
            sim = _MJSimulator(target_fps=target_fps)
        qpos = smpl_params_to_qpos(smpl_params, sim.model)
        breakdowns = simulate_and_score(sim, qpos, fps=target_fps)
        # We emit only contact + dynamic from MuJoCo. Kinematic is
        # supplied externally (caller copies v1 kinematic_feasibility).
        out = {}
        for k, br in breakdowns.items():
            out[f"{k}_feasibility_mj"] = float(br.score)
            for sk, sv in br.sub.items():
                out[f"{k[:3]}_mj_{sk}"] = float(sv)
        return out
    except Exception as e:
        print(f"  rescore_one_cache failed on {cache_path}: {e}")
        return None

# ==========================================================================
# from smpl_feasibility_v3.py
# ==========================================================================

"""v3 SMPL feasibility — v1 semantics, MuJoCo-correct dynamics.

Goal: keep v1's "is this trajectory itself feasible?" interpretation (no
PD smoothing) but compute torques + GRF with **real** articulated-body
inverse dynamics instead of v1's `τ ≈ I·|a|` and `F ≈ m·Cddot` heuristics.

Two MuJoCo calls per frame, both cheap (no integration):
  1. mj_forward  — articulated-body forward kinematics on qpos.
                   Gives body xpos, geom_xpos, subtree COM, contact normals.
  2. mj_inverse  — given (qpos, qvel, qacc), returns qfrc_inverse: the
                   joint-space generalized forces required to produce that
                   motion. The first 6 components (acting on the freejoint)
                   are the implied ground reaction force/torque on the
                   pelvis; the remaining 69 are joint torques.

What this gives us, relative to v1 and MJ-PD:
  * Same literal-trajectory interpretation as v1 (high alignment with humans
    on jitter / impossible-motion artifacts).
  * GRF and torque computed from the actual ABA inertia matrix, gravity,
    and Coriolis terms — not the m·Cddot / I·|a| approximations.
  * Support polygon and COM use MuJoCo's tree-correct quantities.
  * No PD-tracking smoothing, no engine integration → no instability, no
    tuning hyperparameters, ~5× faster than v3's forward-sim variant.

Outputs are emitted as *_v3 keys alongside v1/v2/MJ in the canonical
_eval_results.json (additive merge).
"""



# Reuse the v2-MuJoCo simulator's model loading and qpos conversion, but
# call mj_forward / mj_inverse instead of mj_step.
sys.path.insert(0, os.path.dirname(__file__))


BODY_MASS_DEFAULT = 71.79
SLIP_NORM = 0.05    # m (for tilde v_slip in [0,1])
PEN_NORM  = 0.05    # m


@dataclass
class FeasibilityBreakdown:
    score: float
    sub: Dict[str, float] = field(default_factory=dict)


class _MJKinSim:
    """Minimal wrapper: load PHC's smpl_humanoid.xml, call mj_forward and
    mj_inverse. No actuators used, no integration, no PD.
    """

    def __init__(self, mjcf_path: str = DEFAULT_MJCF):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        # gravity is set in the XML to (0, 0, -9.81). Confirm.
        self.gravity = float(np.linalg.norm(self.model.opt.gravity))
        self.data = mujoco.MjData(self.model)
        self.body_mass = float(self.model.body_mass.sum())

        self.body_id = {}
        for i in range(self.model.nbody):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            self.body_id[n] = i
        self.pelvis_id = self.body_id.get("Pelvis", -1)
        self.foot_body_ids = [self.body_id[b] for b in FOOT_BODIES
                              if b in self.body_id]

        # Per-DoF torque limits indexed by joint name root. Hinges only;
        # the freejoint's 6 dofs use the GRF check, not torque.
        self.dof_torque_limit = np.full(self.model.nv, 100.0)
        self.dof_is_hinge = np.zeros(self.model.nv, dtype=bool)
        for j_id in range(self.model.njnt):
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
            jtype = self.model.jnt_type[j_id]
            dof_adr = self.model.jnt_dofadr[j_id]
            if jtype == mujoco.mjtJoint.mjJNT_FREE:
                # 6 dofs starting at dof_adr; first 3 are translation, last 3 rotation
                continue
            # hinge joint — 1 dof
            self.dof_is_hinge[dof_adr] = True
            root = jname.rsplit("_", 1)[0].split("_", 1)[-1]
            self.dof_torque_limit[dof_adr] = TORQUE_LIMITS.get(root, 100.0)

    def forward(self, qpos: np.ndarray):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.mj.mj_forward(self.model, self.data)

    def required_force(self, qpos: np.ndarray, qvel: np.ndarray,
                        qacc: np.ndarray) -> np.ndarray:
        """Compute the generalized force required to produce qacc at this
        state, i.e. f = M·qacc + qfrc_bias (gravity + Coriolis).

        We avoid mj_inverse because its convention subtracts an internal
        smoothed-bias term — for our purposes (no actuators, no contacts)
        we want the raw required generalized force as if a "magic hand"
        were applying it. This matches the v1 question: what force on
        each DoF is needed to make this trajectory happen?

        Returns:
            f_required: (nv,) array. f[:6] is the force/torque on the
                pelvis freejoint in world frame (linear xyz, angular xyz).
                f[6:] are the per-hinge required torques.
        """
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.qacc[:] = qacc
        self.mj.mj_forward(self.model, self.data)
        # Compute M·qacc via mj_mulM (no need for explicit M).
        Mq = np.zeros(self.model.nv)
        self.mj.mj_mulM(self.model, self.data, Mq, qacc)
        return Mq + self.data.qfrc_bias.copy()

    def com_world(self) -> np.ndarray:
        """Tree-correct world COM via mj_subtreeCom on the root body."""
        # subtree_com[0] is the COM of the entire kinematic tree (the
        # world body's subtree). Populated by mj_forward.
        return self.data.subtree_com[0].copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _point_in_polygon(p: np.ndarray, poly: np.ndarray) -> Tuple[bool, float]:
    """(inside, distance_to_polygon). Empty polygon -> (False, 1.0)."""
    if poly.shape[0] == 0:
        return False, 1.0
    if poly.shape[0] == 1:
        return False, float(np.linalg.norm(p - poly[0]))
    if poly.shape[0] == 2:
        a, b = poly[0], poly[1]
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9),
                          0.0, 1.0))
        return False, float(np.linalg.norm(p - (a + t * ab)))
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(poly)
        verts = poly[hull.vertices]
    except Exception:
        verts = poly
    n = len(verts)
    inside = True; sign = 0
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        s = 1 if cross > 0 else (-1 if cross < 0 else 0)
        if sign == 0:
            sign = s
        elif s != 0 and s != sign:
            inside = False; break
    if inside:
        return True, 0.0
    min_d = min(_seg_dist(p, verts[i], verts[(i + 1) % n])
                for i in range(n))
    return False, float(min_d)


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


# ---------------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------------

def score_trajectory(
    sim: _MJKinSim,
    qpos_traj: np.ndarray,
    *,
    fps: float,
    foot_height_thresh: float = 0.02,
    foot_speed_thresh: float = 0.05,
    rho_lo: float = 0.6,
    rho_hi: float = 1.75,
    met_norm: float = 10000.0,
) -> Dict[str, FeasibilityBreakdown]:
    """Compute v1-style feasibility on qpos_traj using MuJoCo ABA dynamics.

    Returns {kinematic, contact, dynamic, overall} FeasibilityBreakdowns.
    """
    T, nq = qpos_traj.shape
    dt = 1.0 / fps

    # Normalize trajectory z so the lowest foot ever sits at z=0. GVHMR's
    # transl is camera-frame and the floor isn't necessarily at z=0
    # there; without this, all feet read as "in the air" and the contact
    # heuristic fires on nothing. We compute the offset by doing one
    # mj_forward per frame on a copy and finding the min foot body z.
    foot_zs = np.zeros((T, len(sim.foot_body_ids)))
    for t in range(T):
        sim.data.qpos[:] = qpos_traj[t]
        sim.mj.mj_forward(sim.model, sim.data)
        for fi, bid in enumerate(sim.foot_body_ids):
            foot_zs[t, fi] = sim.data.xpos[bid, 2]
    z_offset = float(foot_zs.min())
    qpos_traj = qpos_traj.copy()
    qpos_traj[:, 2] -= z_offset

    # qvel and qacc via finite differences. Pelvis freejoint qvel is
    # linear (3 + 3); for our purposes we use forward differences (clean
    # at start) since we're not stepping an integrator that needs them
    # consistent.
    qvel = np.zeros((T, sim.model.nv))
    qacc = np.zeros((T, sim.model.nv))
    # For non-freejoint hinges, qpos and qvel/qacc index match (1-D).
    # For the freejoint, qpos[0:7] = (x, y, z, qw, qx, qy, qz) but
    # qvel[0:6] = (vx, vy, vz, ωx, ωy, ωz) in world frame. We compute
    # linear velocity for the first 3 dofs and skip angular for now
    # (small relative to translational signal for our use case).
    # Hinge dofs: nv-d index after the first 6.
    hinge_dof_to_qpos = {}
    for j_id in range(sim.model.njnt):
        jtype = sim.model.jnt_type[j_id]
        if jtype == sim.mj.mjtJoint.mjJNT_HINGE:
            dof_adr = sim.model.jnt_dofadr[j_id]
            q_adr = sim.model.jnt_qposadr[j_id]
            hinge_dof_to_qpos[dof_adr] = q_adr
    for t in range(1, T):
        qvel[t, 0:3] = (qpos_traj[t, 0:3] - qpos_traj[t-1, 0:3]) * fps
        for dof_adr, q_adr in hinge_dof_to_qpos.items():
            qvel[t, dof_adr] = (qpos_traj[t, q_adr] -
                                 qpos_traj[t-1, q_adr]) * fps
    for t in range(1, T-1):
        qacc[t] = (qvel[t+1] - qvel[t-1]) * (fps / 2.0)
    qacc[0] = qacc[1]
    qacc[-1] = qacc[-2]

    # ---- per-frame quantities ------------------------------------------
    com_world = np.zeros((T, 3))
    foot_z = np.zeros((T, len(sim.foot_body_ids)))
    foot_pos = np.zeros((T, len(sim.foot_body_ids), 3))
    pelvis_xy = np.zeros((T, 2))
    qfrc_inv = np.zeros((T, sim.model.nv))    # joint-space req force
    for t in range(T):
        qfrc_inv[t] = sim.required_force(qpos_traj[t], qvel[t], qacc[t])
        # required_force already calls mj_forward; reuse that state.
        com_world[t] = sim.com_world()
        for fi, bid in enumerate(sim.foot_body_ids):
            foot_pos[t, fi] = sim.data.xpos[bid]
            foot_z[t, fi]   = sim.data.xpos[bid, 2]
        if sim.pelvis_id >= 0:
            pelvis_xy[t] = sim.data.xpos[sim.pelvis_id, :2]

    # ---- contact heuristic (v1-style): foot close to floor & slow ------
    foot_vel = np.zeros((T, len(sim.foot_body_ids)))
    for fi in range(len(sim.foot_body_ids)):
        for t in range(1, T):
            foot_vel[t, fi] = (np.linalg.norm(
                foot_pos[t, fi] - foot_pos[t-1, fi]) * fps)
    in_contact = (foot_z < foot_height_thresh) & (foot_vel < foot_speed_thresh)
    in_contact_any = in_contact.any(axis=1)

    # ---- s_grf via inverse-dynamics pelvis reaction --------------------
    # qfrc_inverse on the freejoint's 6 dofs is the generalized force on
    # the root in world frame. The first 3 components are the linear
    # reaction force the pelvis "needs" from the floor (the rest of the
    # tree's weight + accel + gravity contribution all rolled in).
    F_pelvis = qfrc_inv[:, 0:3]                  # (T, 3) in world frame
    F_vert = F_pelvis[:, 2]
    F_horiz = np.linalg.norm(F_pelvis[:, :2], axis=-1)
    bw = sim.body_mass * sim.gravity
    v_grf_vert  = float((F_vert > 3.0 * bw).mean())
    v_grf_horiz = float((F_horiz > 0.5 * bw).mean())
    s_grf = 1.0 - 0.5 * (v_grf_vert + v_grf_horiz)

    # NOTE: friction is NOT in the appendix's F_con or F_dyn formula
    # (only mentioned as a diagnostic sub-metric). We don't compute it
    # here — drops one axis that anti-correlated with humans.

    # ---- s_tau via inverse-dynamics joint torques ----------------------
    hinge_dofs = np.where(sim.dof_is_hinge)[0]
    tau_hinge = np.abs(qfrc_inv[:, hinge_dofs])         # (T, n_hinge)
    limits = sim.dof_torque_limit[hinge_dofs]
    over = tau_hinge > limits[None, :]
    v_tau_per_j = over.mean(axis=0)
    s_tau = 1.0 - float(v_tau_per_j.mean())

    # ---- s_met: integrated mechanical effort using ABA torques ---------
    # Joint velocity is qvel at the hinge dofs.
    speed_hinge = np.abs(qvel[:, hinge_dofs])
    MET = float((tau_hinge * speed_hinge).sum() * dt)
    s_met = max(0.0, 1.0 - MET / met_norm)

    # ---- balance: COM in support polygon -------------------------------
    com_xy = com_world[:, :2]
    d_to_support = np.full(T, 1.0)
    for t in range(T):
        contacting = in_contact[t]
        if not contacting.any():
            continue
        poly = foot_pos[t, contacting, :2]
        _, d = _point_in_polygon(com_xy[t], poly)
        d_to_support[t] = d
    v_bal = float(np.mean(np.clip(d_to_support, 0.0, 0.5) / 0.5))
    s_bal = 1.0 - v_bal

    # ---- contact sub-metrics (slip / pen / float) ----------------------
    # Foot slip while in contact
    slip_per_foot = foot_vel * in_contact.astype(float) * dt
    slip_total = float(slip_per_foot.sum() / (2.0 * T))
    v_slip = float(np.clip(slip_total / SLIP_NORM, 0.0, 1.0))

    # Ground penetration
    pen = np.maximum(0.0, -foot_z)
    pen_total = float(pen.sum() / (2.0 * T))
    v_pen = float(np.clip(pen_total / PEN_NORM, 0.0, 1.0))

    # Foot floating: ρ = ‖d/dt(foot - root)‖ / ‖d/dt(root)‖
    root_pos = qpos_traj[:, 0:3]    # pelvis world position
    root_vel = np.zeros_like(root_pos)
    root_vel[1:] = (root_pos[1:] - root_pos[:-1]) * fps
    root_speed = np.linalg.norm(root_vel, axis=-1) + 1e-6
    v_float = 0.0
    for fi in range(len(sim.foot_body_ids)):
        foot_vel_vec = np.zeros((T, 3))
        foot_vel_vec[1:] = (foot_pos[1:, fi] - foot_pos[:-1, fi]) * fps
        rel = foot_vel_vec - root_vel
        rel_speed = np.linalg.norm(rel, axis=-1)
        rho = rel_speed / root_speed
        flagged = (rho < rho_lo) | (rho > rho_hi)
        v_float += float(flagged.mean())
    v_float /= max(1, len(sim.foot_body_ids))

    # ---- F_kin: appendix kinematic recovery (uses joints, not engine) --
    # We deliberately reuse v2's appendix-correct kinematic feasibility
    # here so v3 has a real kin term, not the placeholder we had in MJ.
    # joints / vertices are NOT available from MuJoCo alone — the caller
    # must pass them in (see rescore_one_cache_v3).
    F_kin = FeasibilityBreakdown(score=0.0, sub={})  # placeholder; filled below

    F_con = FeasibilityBreakdown(
        score=float(np.clip(1.0 - 0.25 * (v_slip + v_pen + v_float + v_bal),
                            0.0, 1.0)),
        sub={"v_slip": v_slip, "v_pen": v_pen, "v_float": v_float,
             "v_bal": v_bal,
             "in_contact_any_frac": float(in_contact_any.mean())},
    )
    F_dyn = FeasibilityBreakdown(
        score=float(np.clip((s_tau + s_grf + s_met + s_bal) / 4.0,
                            0.0, 1.0)),
        sub={"s_tau": s_tau, "s_grf": s_grf, "s_met": s_met,
             "s_bal": s_bal,
             "v_grf_vert": v_grf_vert, "v_grf_horiz": v_grf_horiz,
             "MET": MET},
    )
    return {
        "kinematic": F_kin,
        "contact":   F_con,
        "dynamic":   F_dyn,
    }


# ---------------------------------------------------------------------------
# Driver-level convenience
# ---------------------------------------------------------------------------

def rescore_one_cache(
    cache_path: str,
    sim: Optional[_MJKinSim] = None,
    body_model = None,
    *,
    target_fps: float = 30.0,
) -> Optional[Dict[str, float]]:
    """Load a GVHMR cache, score with v3.

    Also runs v2's kinematic_feasibility on the recovered joints/vertices
    so kin_v3 := kin_v2 (the appendix's kinematic axis is well-defined
    only in joint-space; MuJoCo doesn't add information there).
    """
    import torch as _t
    try:
        blob = _t.load(cache_path, weights_only=False, map_location="cpu")
        pred = blob.get("pred", blob)
        smpl_params = pred["smpl_params_global"]
        if smpl_params["body_pose"].shape[0] < 2:
            return None
        if sim is None:
            sim = _MJKinSim()
        qpos = smpl_params_to_qpos(smpl_params, sim.model)
        breakdowns = score_trajectory(sim, qpos, fps=target_fps)

        # Optional: pair with v2 kinematic if body_model passed in.
        if body_model is not None:
            from astrolabe.scorers.video.smpl_feasibility_v2 import (
                kinematic_feasibility_v2,
            )
            T = smpl_params["body_pose"].shape[0]
            betas = smpl_params["betas"]
            if betas.ndim == 1: betas = betas.unsqueeze(0).expand(T, -1)
            elif betas.shape[0] == 1: betas = betas.expand(T, -1)
            with _t.no_grad():
                out = body_model(
                    betas=betas.float(),
                    global_orient=smpl_params["global_orient"].float(),
                    body_pose=smpl_params["body_pose"].float(),
                    transl=smpl_params["transl"].float(),
                    left_hand_pose=_t.zeros(T, 45),
                    right_hand_pose=_t.zeros(T, 45),
                    jaw_pose=_t.zeros(T, 3),
                    leye_pose=_t.zeros(T, 3),
                    reye_pose=_t.zeros(T, 3),
                    expression=_t.zeros(T, 10),
                    return_full_pose=True,
                )
            verts = out.vertices.detach().cpu().numpy()
            joints = out.joints.detach().cpu().numpy()
            rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
            verts = verts @ rot.T
            joints = joints @ rot.T
            z0 = verts[0, :, 2].min()
            verts[:, :, 2] -= z0; joints[:, :, 2] -= z0
            faces = body_model.faces
            if isinstance(faces, _t.Tensor):
                faces = faces.detach().cpu().numpy()
            kin = kinematic_feasibility_v2(joints, verts, faces, target_fps)
            breakdowns["kinematic"] = FeasibilityBreakdown(
                score=kin.score, sub=dict(kin.sub),
            )

        overall = float(np.mean([breakdowns[k].score
                                   for k in ("kinematic", "contact", "dynamic")]))
        out = {}
        for k, br in breakdowns.items():
            out[f"{k}_feasibility_v3"] = float(br.score)
            for sk, sv in br.sub.items():
                out[f"{k[:3]}_v3_{sk}"] = float(sv)
        out["overall_feasibility_v3"] = overall
        return out
    except Exception as e:
        print(f"  rescore_one_cache_v3 failed on {cache_path}: {e}")
        return None

