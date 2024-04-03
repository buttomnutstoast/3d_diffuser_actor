from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import einops
import torch
from torch import nn
from torch.nn import functional as F

from diffuser_actor.utils.layers import ParallelAttention
from diffuser_actor.utils.encoder import Encoder
from diffuser_actor.utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat
)
import utils.pytorch3d_transforms as pytorch3d_transforms


class DiffusionPlanner(nn.Module):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 output_dim=7,
                 num_vis_ins_attn_layers=2,
                 num_query_cross_attn_layers=8,
                 use_instruction=False,
                 use_goal=False,
                 use_goal_at_test=True,
                 feat_scales_to_use=1,
                 attn_rounds=1,
                 weight_tying=False,
                 gripper_loc_bounds=None,
                 rotation_parametrization='quat',
                 diffusion_timesteps=100):
        super().__init__()
        self._use_goal = use_goal
        self._use_goal_at_test = use_goal_at_test
        self._rotation_parametrization = rotation_parametrization
        self.prediction_head = DiffusionHead(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            num_vis_ins_attn_layers=num_vis_ins_attn_layers,
            num_query_cross_attn_layers=num_query_cross_attn_layers,
            use_instruction=use_instruction,
            use_goal=use_goal,
            feat_scales_to_use=feat_scales_to_use,
            attn_rounds=attn_rounds,
            weight_tying=weight_tying,
            rotation_parametrization=rotation_parametrization
        )
        self.position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="sample"
        )
        self.rotation_noise_scheduler = DDPMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample"
        )
        self.n_steps = diffusion_timesteps
        self.gripper_loc_bounds = torch.tensor(gripper_loc_bounds)

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (
            trajectory_mask,
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper,
            goal_gripper
        ) = fixed_inputs

        return self.prediction_head(
            trajectory,
            trajectory_mask,
            timestep,
            visible_rgb=rgb_obs,
            visible_pcd=pcd_obs,
            curr_gripper=curr_gripper,
            goal_gripper=goal_gripper,
            instruction=instruction
        )

    def conditional_sample(self, condition_data, condition_mask, fixed_inputs):
        self.position_noise_scheduler.set_timesteps(self.n_steps)
        self.rotation_noise_scheduler.set_timesteps(self.n_steps)

        # Random trajectory, conditioned on start-end
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device
        )
        trajectory = trajectory + condition_data

        # Iterative denoising
        timesteps = self.position_noise_scheduler.timesteps
        for t in timesteps:
            out = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(trajectory.device).long(),
                fixed_inputs
            )
            out = out[-1]  # keep only last layer's output
            out[condition_mask] = condition_data[condition_mask]
            if t == timesteps[-1]:
                trajectory = out
            else:
                pos = self.position_noise_scheduler.step(
                    out[..., :3], t, trajectory[..., :3]
                ).prev_sample
                rot = self.rotation_noise_scheduler.step(
                    out[..., 3:9], t, trajectory[..., 3:9]
                ).prev_sample
                trajectory = torch.cat((pos, rot), -1)

        return trajectory

    def compute_trajectory(
        self,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        goal_gripper
    ):
        # Normalize all pos
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        goal_gripper = goal_gripper.clone()
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[:, :3] = self.normalize_pos(curr_gripper[:, :3])
        goal_gripper[:, :3] = self.normalize_pos(goal_gripper[:, :3])
        curr_gripper = self.convert_rot(curr_gripper)
        goal_gripper = self.convert_rot(goal_gripper)

        # Prepare inputs
        fixed_inputs = (
            trajectory_mask,
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper,
            goal_gripper
        )

        # Condition on start-end pose
        B, D = curr_gripper.shape
        cond_data = torch.zeros(
            (B, trajectory_mask.size(1), D),
            device=rgb_obs.device
        )
        cond_mask = torch.zeros_like(cond_data)
        # start pose
        cond_data[:, 0] = curr_gripper
        cond_mask[:, 0] = 1
        # end pose
        if self._use_goal_at_test:
            for d in range(len(cond_data)):
                neg_len_ = -trajectory_mask[d].sum().long()
                cond_data[d][neg_len_ - 1] = goal_gripper[d]
                cond_mask[d][neg_len_ - 1:] = 1
        cond_mask = cond_mask.bool()

        # Sample
        trajectory = self.conditional_sample(
            cond_data,
            cond_mask,
            fixed_inputs
        )

        # Normalize quaternion
        if self._rotation_parametrization != '6D':
            trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
        # Back to quaternion
        trajectory = self.unconvert_rot(trajectory)
        # unnormalize position
        trajectory[:, :, :3] = self.unnormalize_pos(trajectory[:, :, :3])

        return trajectory

    def normalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        pos_min = self.gripper_loc_bounds[0].float().to(pos.device)
        pos_max = self.gripper_loc_bounds[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

    def convert_rot(self, signal):
        signal[..., 3:7] = normalise_quat(signal[..., 3:7])
        if self._rotation_parametrization == '6D':
            rot = pytorch3d_transforms.quaternion_to_matrix(signal[..., 3:7])
            res = signal[..., 7:] if signal.size(-1) > 7 else None
            if len(rot.shape) == 4:
                B, L, D1, D2 = rot.shape
                rot = rot.reshape(B * L, D1, D2)
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
                rot_6d = rot_6d.reshape(B, L, 6)
            else:
                rot_6d = get_ortho6d_from_rotation_matrix(rot)
            signal = torch.cat([signal[..., :3], rot_6d], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        if self._rotation_parametrization == '6D':
            res = signal[..., 9:] if signal.size(-1) > 9 else None
            if len(signal.shape) == 3:
                B, L, _ = signal.shape
                rot = signal[..., 3:9].reshape(B * L, 6)
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = pytorch3d_transforms.matrix_to_quaternion(mat)
                quat = quat.reshape(B, L, 4)
            else:
                rot = signal[..., 3:9]
                mat = compute_rotation_matrix_from_ortho6d(rot)
                quat = pytorch3d_transforms.matrix_to_quaternion(mat)
            signal = torch.cat([signal[..., :3], quat], dim=-1)
            if res is not None:
                signal = torch.cat((signal, res), -1)
        return signal

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb_obs,
        pcd_obs,
        instruction,
        curr_gripper,
        goal_gripper,
        run_inference=False
    ):
        # gt_trajectory is expected to be in the quaternion format
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask,
                rgb_obs,
                pcd_obs,
                instruction,
                curr_gripper,
                goal_gripper
            )
        # Normalize all pos
        gt_trajectory = gt_trajectory.clone()
        pcd_obs = pcd_obs.clone()
        curr_gripper = curr_gripper.clone()
        goal_gripper = goal_gripper.clone()
        gt_trajectory[:, :, :3] = self.normalize_pos(gt_trajectory[:, :, :3])
        pcd_obs = torch.permute(self.normalize_pos(
            torch.permute(pcd_obs, [0, 1, 3, 4, 2])
        ), [0, 1, 4, 2, 3])
        curr_gripper[:, :3] = self.normalize_pos(curr_gripper[:, :3])
        goal_gripper[:, :3] = self.normalize_pos(goal_gripper[:, :3])

        # Convert rotation parametrization
        gt_trajectory = self.convert_rot(gt_trajectory)
        curr_gripper = self.convert_rot(curr_gripper)
        goal_gripper = self.convert_rot(goal_gripper)

        # Prepare inputs
        fixed_inputs = (
            trajectory_mask,
            rgb_obs,
            pcd_obs,
            instruction,
            curr_gripper,
            goal_gripper
        )

        # Condition on start-end pose
        cond_data = torch.zeros_like(gt_trajectory)
        cond_mask = torch.zeros_like(cond_data)
        cond_mask = cond_mask.bool()

        # Sample noise
        noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

        # Sample a random timestep
        timesteps = torch.randint(
            0,
            self.position_noise_scheduler.config.num_train_timesteps,
            (len(noise),), device=noise.device
        ).long()

        # Add noise to the clean trajectories
        pos = self.position_noise_scheduler.add_noise(
            gt_trajectory[..., :3], noise[..., :3],
            timesteps
        )
        rot = self.rotation_noise_scheduler.add_noise(
            gt_trajectory[..., 3:9], noise[..., 3:9],
            timesteps
        )
        noisy_trajectory = torch.cat((pos, rot), -1)
        noisy_trajectory[cond_mask] = cond_data[cond_mask]  # condition

        # Predict the noise residual
        pred = self.policy_forward_pass(
            noisy_trajectory, timesteps,
            fixed_inputs
        )
        target = gt_trajectory

        # Compute loss
        total_loss = 0
        for layer_pred in pred:
            trans = layer_pred[..., :3]
            rot = layer_pred[..., 3:9]
            loss = (
                100 * F.l1_loss(trans, target[..., :3], reduction='mean')
                + 10 * F.l1_loss(rot, target[..., 3:9], reduction='mean')
            )
            total_loss = total_loss + loss
        return total_loss


class DiffusionHead(Encoder):

    def __init__(self,
                 backbone="clip",
                 image_size=(256, 256),
                 embedding_dim=60,
                 output_dim=7,
                 num_attn_heads=8,
                 num_vis_ins_attn_layers=2,
                 num_query_cross_attn_layers=6,
                 use_instruction=False,
                 use_goal=False,
                 use_sigma=False,
                 feat_scales_to_use=1,
                 attn_rounds=1,
                 weight_tying=False,
                 rotation_parametrization='quat'):
        super().__init__(
            backbone=backbone,
            image_size=image_size,
            embedding_dim=embedding_dim,
            num_sampling_level=feat_scales_to_use,
            use_sigma=use_sigma
        )
        self.use_instruction = use_instruction
        self.use_goal = use_goal
        self.attn_rounds = attn_rounds
        self.feat_scales = feat_scales_to_use
        self.rotation_parametrization = rotation_parametrization
        if self.rotation_parametrization == '6D':
            output_dim += 2

        # Encoders
        self.traj_encoder = nn.Sequential(
            nn.Linear(9, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_encoder = nn.Linear(output_dim, embedding_dim)
        if use_goal:
            self.goal_gripper_encoder = nn.Linear(output_dim, embedding_dim)

        # Attention from vision to language
        if use_instruction and weight_tying:
            layer = ParallelAttention(
                num_layers=num_vis_ins_attn_layers,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False
            )
            self.vl_attention = nn.ModuleList([
                layer
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
        elif use_instruction:
            self.vl_attention = nn.ModuleList([
                ParallelAttention(
                    num_layers=num_vis_ins_attn_layers,
                    d_model=embedding_dim, n_heads=num_attn_heads,
                    self_attention1=False, self_attention2=False,
                    cross_attention1=True, cross_attention2=False
                )
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])

        # Attention from trajectory queries to language
        if weight_tying:
            layer = ParallelAttention(
                num_layers=1,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=False, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=False, apply_ffn=False
            )
            self.traj_lang_attention = nn.ModuleList([
                layer
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
        else:
            self.traj_lang_attention = nn.ModuleList([
                ParallelAttention(
                    num_layers=1,
                    d_model=embedding_dim, n_heads=num_attn_heads,
                    self_attention1=False, self_attention2=False,
                    cross_attention1=True, cross_attention2=False,
                    rotary_pe=False, apply_ffn=False
                )
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])

        # Attention from trajectory queries to context
        if weight_tying:
            layer = ParallelAttention(
                num_layers=num_query_cross_attn_layers - 2,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=True, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=True, use_adaln=True
            )
            self.traj_attention = nn.ModuleList([
                layer
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
            layer = ParallelAttention(
                num_layers=2,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=True, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=True, use_adaln=True
            )
            self.pos_attention = nn.ModuleList([
                layer
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
            layer = ParallelAttention(
                num_layers=2,
                d_model=embedding_dim, n_heads=num_attn_heads,
                self_attention1=True, self_attention2=False,
                cross_attention1=True, cross_attention2=False,
                rotary_pe=True, use_adaln=True
            )
            self.rot_attention = nn.ModuleList([
                layer
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
        else:
            self.traj_attention = nn.ModuleList([
                ParallelAttention(
                    num_layers=num_query_cross_attn_layers - 2,
                    d_model=embedding_dim, n_heads=num_attn_heads,
                    self_attention1=True, self_attention2=False,
                    cross_attention1=True, cross_attention2=False,
                    rotary_pe=True, use_adaln=True
                )
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
            self.pos_attention = nn.ModuleList([
                ParallelAttention(
                    num_layers=2,
                    d_model=embedding_dim, n_heads=num_attn_heads,
                    self_attention1=True, self_attention2=False,
                    cross_attention1=True, cross_attention2=False,
                    rotary_pe=True, use_adaln=True
                )
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])
            self.rot_attention = nn.ModuleList([
                ParallelAttention(
                    num_layers=2,
                    d_model=embedding_dim, n_heads=num_attn_heads,
                    self_attention1=True, self_attention2=False,
                    cross_attention1=True, cross_attention2=False,
                    rotary_pe=True, use_adaln=True
                )
                for _ in range(self.attn_rounds)
                for _ in range(self.feat_scales)
            ])

        # Regression after every attention to a scale
        self.pos_regressor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim, 3)
            )
            for _ in range(self.attn_rounds)
            for _ in range(self.feat_scales)
        ])
        self.rot_regressor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim, output_dim - 3)
            )
            for _ in range(self.attn_rounds)
            for _ in range(self.feat_scales)
        ])

    def forward(self, trajectory, trajectory_mask, timestep,
                visible_rgb, visible_pcd, curr_gripper, goal_gripper,
                instruction):
        """
        Arguments:
            trajectory: (B, trajectory_length, 3+6+X)
            trajectory_mask: (B, trajectory_length)
            timestep: (B, 1)
            visible_rgb: (B, num_cameras, 3, H, W) in [0, 1]
            visible_pcd: (B, num_cameras, 3, H, W) in world coordinates
            curr_gripper: (B, output_dim)
            goal_gripper: (B, output_dim)
            instruction: (B, max_instruction_length, 512)
        """
        # Trajectory features
        traj_feats = self.traj_encoder(trajectory)  # (B, L, F)
        traj_pos = self.relative_pe_layer(trajectory[..., :3])

        # Timestep features (B, 1, F)
        time_feats, time_pos = self.encode_denoising_timestep(timestep)

        # Compute visual features/positional embeddings at different scales
        rgb_feats_pyramid, pcd_pyramid = self.encode_images(
            visible_rgb, visible_pcd
        )

        # Encode instruction (B, 53, F)
        instr_feats, instr_pos = None, None
        if self.use_instruction:
            instr_feats, instr_pos = self.encode_instruction(instruction)

        # Encode current gripper (B, 1, F)
        curr_gripper_feats = self.curr_gripper_encoder(curr_gripper)
        curr_gripper_feats = curr_gripper_feats[:, None]
        curr_gripper_embs, curr_gripper_pos = self.encode_curr_gripper(
            curr_gripper, batch_size=len(traj_feats)
        )
        curr_gripper_feats = curr_gripper_feats + curr_gripper_embs

        # Encode goal gripper (B, 1, F)
        goal_gripper_feats, goal_gripper_pos = None, None
        if self.use_goal:
            goal_gripper_embs, goal_gripper_pos = self.encode_goal_gripper(
                goal_gripper, batch_size=len(traj_feats)
            )
            goal_gripper_feats = self.goal_gripper_encoder(goal_gripper)
            goal_gripper_feats = goal_gripper_feats[:, None]
            goal_gripper_feats = goal_gripper_feats + goal_gripper_embs

        # Attention layers
        n_trajectory = []
        for attn_round in range(self.attn_rounds):
            for scale in range(self.feat_scales):
                # Local attention
                p_inds = None

                # One attention iteration
                update = self._one_attention_round(
                    rgb_feats_pyramid, pcd_pyramid,  # visual
                    instr_feats, instr_pos,  # language
                    curr_gripper_feats, curr_gripper_pos,  # current gripper
                    goal_gripper_feats, goal_gripper_pos,  # goal gripper
                    time_feats, time_pos,  # time
                    traj_feats, traj_pos, trajectory_mask,  # trajectory
                    attn_round, scale, p_inds
                )
                trajectory = torch.cat((
                    trajectory[..., :3] + update[..., :3],
                    update[..., 3:]
                ), -1)
                n_trajectory.append(trajectory)

        return n_trajectory

    def _one_attention_round(
        self,
        rgb_feats_pyramid, pcd_pyramid,  # visual
        instr_feats, instr_pos,  # language
        curr_gripper_feats, curr_gripper_pos,  # current gripper
        goal_gripper_feats, goal_gripper_pos,  # goal gripper
        time_feats, time_pos,  # time
        traj_feats, traj_pos, trajectory_mask,  # trajectory
        attn_round, scale, p_inds=None
    ):
        # Visual context
        context_feats = einops.rearrange(
            rgb_feats_pyramid[scale],
            "b ncam c h w -> b (ncam h w) c"
        )
        context_pos = pcd_pyramid[scale]
        if p_inds is not None:
            context_feats = torch.stack([
                f[i]  # (nn, c)
                for f, i in zip(context_feats, p_inds)
            ])
            context_pos = torch.stack([
                f[i] for f, i in zip(context_pos, p_inds)
            ])
        context_pos = self.relative_pe_layer(context_pos)

        # Language context
        if self.use_instruction:
            # Attention from vision to language
            l_offset = attn_round * self.feat_scales + scale
            context_feats, _ = self.vl_attention[l_offset](
                seq1=context_feats, seq1_key_padding_mask=None,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=None, seq2_sem_pos=None
            )

        # Concatenate rest of context (gripper)
        context_feats = torch.cat([context_feats, curr_gripper_feats], dim=1)
        context_pos = torch.cat([context_pos, curr_gripper_pos], dim=1)

        # Concatenate goal gripper if used
        if self.use_goal:
            context_feats = torch.cat([context_feats, goal_gripper_feats], 1)
            context_pos = torch.cat([context_pos, goal_gripper_pos], 1)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        l_offset = attn_round * self.feat_scales + scale
        if self.use_instruction:
            traj_feats, _ = self.traj_lang_attention[l_offset](
                seq1=traj_feats, seq1_key_padding_mask=trajectory_mask,
                seq2=instr_feats, seq2_key_padding_mask=None,
                seq1_pos=None, seq2_pos=None,
                seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
            )
        traj_feats, _ = self.traj_attention[l_offset](
            seq1=traj_feats, seq1_key_padding_mask=trajectory_mask,
            seq2=context_feats, seq2_key_padding_mask=None,
            seq1_pos=traj_pos, seq2_pos=context_pos,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None,
            ada_sgnl=time_feats.squeeze(1)
        )
        pos_feats, _ = self.pos_attention[l_offset](
            seq1=traj_feats, seq1_key_padding_mask=trajectory_mask,
            seq2=context_feats, seq2_key_padding_mask=None,
            seq1_pos=traj_pos, seq2_pos=context_pos,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None,
            ada_sgnl=time_feats.squeeze(1)
        )
        rot_feats, _ = self.rot_attention[l_offset](
            seq1=traj_feats, seq1_key_padding_mask=trajectory_mask,
            seq2=context_feats, seq2_key_padding_mask=None,
            seq1_pos=traj_pos, seq2_pos=context_pos,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None,
            ada_sgnl=time_feats.squeeze(1)
        )

        # Regress trajectory
        return torch.cat((
            self.pos_regressor[l_offset](pos_feats),
            self.rot_regressor[l_offset](rot_feats)
        ), -1)  # (B, L, output_dim)
