"""Main script for trajectory optimization."""

import os
import random

import numpy as np
import torch
import torch.distributed as dist

from diffuser_actor.trajectory_optimization.chained_diffuser import (
    DiffusionPlanner
)

from utils.common_utils import count_parameters, get_gripper_loc_bounds
from main_trajectory import Arguments, generate_visualizations, traj_collate_fn
from main_trajectory import TrainTester as TrajTrainTester


class TrainTester(TrajTrainTester):
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = DiffusionPlanner(
            backbone=self.args.backbone,
            image_size=tuple(int(x) for x in self.args.image_size.split(",")),
            embedding_dim=self.args.embedding_dim,
            output_dim=7,
            num_vis_ins_attn_layers=self.args.num_vis_ins_attn_layers,
            use_instruction=bool(self.args.use_instruction),
            use_goal=True,
            weight_tying=True,
            gripper_loc_bounds=self.args.gripper_loc_bounds,
            rotation_parametrization='6D',
            diffusion_timesteps=self.args.diffusion_timesteps
        )
        print("Model parameters:", count_parameters(_model))

        return _model

    def train_one_step(self, model, criterion, optimizer, step_id, sample):
        """Run a single training step."""
        if step_id % self.args.accumulate_grad_batches == 0:
            optimizer.zero_grad()

        if self.args.keypose_only:
            sample["trajectory"] = sample["trajectory"][:, [-1]]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
        else:
            sample["trajectory"] = sample["trajectory"][:, 1:]
            sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]

        # Forward pass
        curr_gripper = (
            sample["curr_gripper"] if self.args.num_history < 1
            else sample["curr_gripper_history"][:, -self.args.num_history:]
        )
        out = model(
            sample["trajectory"],
            sample["trajectory_mask"],
            sample["rgbs"],
            sample["pcds"],
            sample["instr"],
            curr_gripper,
            sample["action"]
        )

        # Backward pass
        loss = criterion.compute_loss(out)
        loss.backward()

        # Update
        if step_id % self.args.accumulate_grad_batches == self.args.accumulate_grad_batches - 1:
            optimizer.step()

        # Log
        if dist.get_rank() == 0 and (step_id + 1) % self.args.val_freq == 0:
            self.writer.add_scalar("lr", self.args.lr, step_id)
            self.writer.add_scalar("train-loss/noise_mse", loss, step_id)

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in enumerate(loader):
            if i == val_iters:
                break

            if self.args.keypose_only:
                sample["trajectory"] = sample["trajectory"][:, [-1]]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
            else:
                sample["trajectory"] = sample["trajectory"][:, 1:]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]

            curr_gripper = (
                sample["curr_gripper"] if self.args.num_history < 1
                else sample["curr_gripper_history"][:, -self.args.num_history:]
            )
            action = model(
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device),
                sample["rgbs"].to(device),
                sample["pcds"].to(device),
                sample["instr"].to(device),
                curr_gripper.to(device),
                sample["action"].to(device),
                run_inference=True
            )
            losses, losses_B = criterion.compute_metrics(
                action,
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device)
            )

            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            # Gather per-task statistics
            tasks = np.array(sample["task"])
            for n, l in losses_B.items():
                for task in np.unique(tasks):
                    key = f"{split}-loss/{task}/{n}"
                    l_task = l[tasks == task].mean()
                    if key not in values:
                        values[key] = torch.Tensor([]).to(device)
                    values[key] = torch.cat([values[key], l_task.unsqueeze(0)])

            # Generate visualizations
            if i == 0 and dist.get_rank() == 0 and step_id > -1:
                viz_key = f'{split}-viz/viz'
                viz = generate_visualizations(
                    action,
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device)
                )
                self.writer.add_image(viz_key, viz, step_id)

        # Log all statistics
        values = self.synchronize_between_processes(values)
        values = {k: v.mean().item() for k, v in values.items()}
        if dist.get_rank() == 0:
            if step_id > -1:
                for key, val in values.items():
                    self.writer.add_scalar(key, val, step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return values.get('val-losses/traj_pos_acc_001', None)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Arguments
    args = Arguments().parse_args()
    print("Arguments:")
    print(args)
    print("-" * 100)
    if args.gripper_loc_bounds is None:
        args.gripper_loc_bounds = np.array([[-2, -2, -2], [2, 2, 2]]) * 1.0
    else:
        args.gripper_loc_bounds = get_gripper_loc_bounds(
            args.gripper_loc_bounds,
            task=args.tasks[0] if len(args.tasks) == 1 else None,
            buffer=args.gripper_loc_bounds_buffer,
        )
    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count", torch.cuda.device_count())
    args.local_rank = int(os.environ["LOCAL_RANK"])

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Run
    train_tester = TrainTester(args)
    train_tester.main(collate_fn=traj_collate_fn)
