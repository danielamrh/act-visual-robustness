"""Command builders for the LeRobot CLI (pinned to lerobot 0.6.1)."""

from __future__ import annotations

ALOHA_TASKS = {
    "transfer_cube": "AlohaTransferCube-v0",
    "insertion": "AlohaInsertion-v0",
}


def _flag(value: bool) -> str:
    return "true" if value else "false"


def eval_cmd(
    policy_path: str,
    output_dir: str,
    task: str = "AlohaTransferCube-v0",
    n_episodes: int = 50,
    # Envs step sequentially anyway, so a small batch costs no time. LeRobot
    # keeps every rendered frame of the first batches for videos: 25 envs x 400
    # steps x 640x480x3 is ~9 GB and gets the process OOM-killed on Colab.
    batch_size: int = 5,
    device: str = "cuda",
) -> list[str]:
    return [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        f"--policy.device={device}",
        "--env.type=aloha",
        f"--env.task={task}",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={batch_size}",
        # Async workers start fresh on Python 3.13 and never import gym_aloha,
        # so its envs are unregistered there. Colab Free has 2 cores anyway.
        "--eval.use_async_envs=false",
        f"--output_dir={output_dir}",
    ]


def train_cmd(
    output_dir: str,
    dataset: str = "lerobot/aloha_sim_transfer_cube_human",
    task: str = "AlohaTransferCube-v0",
    steps: int = 80_000,
    batch_size: int = 8,
    save_freq: int = 5_000,
    log_freq: int = 200,
    env_eval_freq: int = 20_000,
    eval_episodes: int = 10,
    eval_batch_size: int = 5,
    num_workers: int = 2,
    seed: int = 1000,
    use_amp: bool = False,
    save_checkpoint: bool = True,
    wandb: bool = False,
    extra: list[str] | None = None,
) -> list[str]:
    """Fresh ACT training run. `env_eval_freq=0` disables in-training rollouts."""
    return [
        "lerobot-train",
        "--policy.type=act",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--policy.use_amp={_flag(use_amp)}",
        f"--dataset.repo_id={dataset}",
        "--env.type=aloha",
        f"--env.task={task}",
        f"--output_dir={output_dir}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--save_freq={save_freq}",
        f"--save_checkpoint={_flag(save_checkpoint)}",
        f"--log_freq={log_freq}",
        f"--env_eval_freq={env_eval_freq}",
        f"--eval.n_episodes={eval_episodes}",
        f"--eval.batch_size={eval_batch_size}",
        "--eval.use_async_envs=false",
        f"--num_workers={num_workers}",
        f"--seed={seed}",
        f"--wandb.enable={_flag(wandb)}",
        *(extra or []),
    ]


def resume_cmd(config_path: str) -> list[str]:
    """Continue a run from `<run>/checkpoints/last/pretrained_model/train_config.json`."""
    return ["lerobot-train", f"--config_path={config_path}", "--resume=true"]
