"""Generate the Colab notebooks in notebooks/ (source of truth for their cells).

    python tools/build_notebooks.py
"""

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
GH = "danielamrh/act-visual-robustness"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def badge(name):
    return (f'<a href="https://colab.research.google.com/github/{GH}/blob/main/notebooks/{name}.ipynb">'
            '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>')


SETUP = [
    md("## 0 · Environment check"),
    code("""
import sys, subprocess
print(sys.version)
assert sys.version_info >= (3, 12), "LeRobot 0.6.x needs Python >= 3.12"
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout or "No GPU - switch the runtime to T4!")
"""),
    md("## 1 · Drive + repo + install\nCode comes from GitHub (always fresh via `git pull`), outputs go to Google Drive."),
    code("""
import os
from google.colab import drive
drive.mount("/content/drive")

REPO = "act-visual-robustness"
REPO_DIR = f"/content/{REPO}"
if not os.path.exists(REPO_DIR):
    !git clone -q https://github.com/danielamrh/{REPO}.git {REPO_DIR}
else:
    !git -C {REPO_DIR} pull -q
%cd {REPO_DIR}
COMMIT = !git rev-parse --short HEAD
COMMIT = COMMIT[0]
print("commit:", COMMIT)

# labmaze (pulled in by dm-control) has no wheel for Python >= 3.13 and needs Bazel
# to build. gym-aloha never uses it, so install an empty placeholder first.
if sys.version_info >= (3, 13):
    !pip install -q ./tools/labmaze_stub

# installs lerobot[aloha] pinned in pyproject.toml (takes a few minutes)
!pip install -q -e ".[sim]"
sys.path.insert(0, f"{REPO_DIR}/src")  # editable install is only picked up after a restart

# GPU rendering: without NVIDIA's EGL registration MuJoCo silently renders on the CPU
from avr.colab import ensure_nvidia_egl, gl_renderer
ensure_nvidia_egl()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
renderer = gl_renderer()
print("OpenGL renderer:", renderer)
if "llvmpipe" in renderer.lower() or renderer.startswith("failed"):
    print("⚠ MuJoCo is NOT rendering on the GPU - rollouts will be extremely slow")

DRIVE_ROOT = "/content/drive/MyDrive/act_robustness"
os.makedirs(DRIVE_ROOT, exist_ok=True)
"""),
    md("### Quick check: can we create and render the ALOHA env?"),
    code("""
import gymnasium as gym
import gym_aloha  # registers gym_aloha/* envs
import matplotlib.pyplot as plt

import time
env = gym.make("gym_aloha/AlohaTransferCube-v0")
obs, _ = env.reset(seed=0)
img = obs["top"] if isinstance(obs, dict) and "top" in obs else env.render()

# speed check: env step incl. observation rendering, zero actions
t0 = time.perf_counter()
for _ in range(50):
    env.step(env.action_space.sample() * 0)
dt = (time.perf_counter() - t0) / 50
env.close()
print(f"{dt*1000:.0f} ms per env step -> one 400-step episode ≈ {400*dt:.0f} s (+ video rendering)")
plt.imshow(img); plt.axis("off"); plt.title("AlohaTransferCube-v0, seed 0");
"""),
]


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


# ---------------------------------------------------------------- notebook 01
nb01 = [
    md(f"""
# 01 · Evaluate the pretrained ACT checkpoint (Step 1b)

{badge("01_eval_pretrained")}

Sanity check for the evaluation pipeline before training anything ourselves.
The Hugging Face checkpoint [`lerobot/act_aloha_sim_transfer_cube_human`](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human)
reports **83.0 % success over 500 episodes** on `AlohaTransferCube-v0`.
If our number lands inside that ballpark (mind the confidence interval), the pipeline is trustworthy.

**Runtime → Change runtime type → T4 GPU** before running.
"""),
    *SETUP,
    md("""
## 2 · Migrate the checkpoint to the current LeRobot format
The Hub checkpoint predates LeRobot's processor pipelines (normalization used to live inside the model).
LeRobot ships a migration script for exactly this; we run it once and cache the result on Drive.
"""),
    code("""
import lerobot, pathlib
print("lerobot", lerobot.__version__)

PRETRAINED = "lerobot/act_aloha_sim_transfer_cube_human"
POLICY_DIR = f"{DRIVE_ROOT}/checkpoints/act_aloha_sim_transfer_cube_human_migrated"
migrate_script = pathlib.Path(lerobot.__file__).parent / "processor" / "migrate_policy_normalization.py"

if not os.path.exists(f"{POLICY_DIR}/config.json"):
    !python {migrate_script} --pretrained-path {PRETRAINED} --output-dir {POLICY_DIR}
!ls {POLICY_DIR}
"""),
    md("""
## 3 · Smoke test (10 episodes)
Quick check that env, rendering and policy work together before spending GPU time.

Evaluations run as **background jobs** that only write to a log file; the cell prints one status line
per interval (elapsed time, free RAM, the job's RAM, latest progress). Streaming LeRobot's progress bars
into the cell crashed the browser tab on long runs.
"""),
    code("""
import time
from avr.eval.chunked import run_chunked_eval

EVAL_BASE = f"{DRIVE_ROOT}/eval/pretrained_transfer_cube"
smoke_root = f"{EVAL_BASE}/smoke_{COMMIT}_{time.strftime('%Y%m%d-%H%M%S')}"
run_chunked_eval(POLICY_DIR, smoke_root, n_episodes=10, chunk_size=10, interval=15);
"""),
    md("""
## 4 · Full evaluation (500 episodes, resumable)
500 episodes give a ±~3.5 % confidence interval, like the model card. They run in **chunks of 100**,
each a separate process with its own seed range (1000-1099, 1100-1199, ...). Finished chunks stay on Drive:
if the session dies, re-run sections 0–2 and this cell, and it continues with the missing chunks.
Each chunk takes roughly 10 minutes.
"""),
    code("""
N_EPISODES = 500
EVAL_ROOT = f"{EVAL_BASE}/n{N_EPISODES}_seed1000"
info_paths = run_chunked_eval(POLICY_DIR, EVAL_ROOT, n_episodes=N_EPISODES, chunk_size=100, seed=1000)
"""),
    md("## 5 · Summary: success rate, confidence interval, failure stages\nWorks on partial results too (e.g. 3 of 5 chunks done)."),
    code("""
import glob, json
from avr.eval.chunked import chunk_info_paths
from avr.eval.stats import merge_eval_infos, summarize_eval, format_summary

EVAL_ROOT = f"{EVAL_BASE}/n500_seed1000"
info_paths = chunk_info_paths(EVAL_ROOT)
print(f"{len(info_paths)} finished chunks in {EVAL_ROOT}")
summary = summarize_eval(merge_eval_infos(info_paths))
print(format_summary(summary))
print("\\nReference (model card): 83.0 % over 500 episodes")

with open(f"{EVAL_ROOT}/summary.json", "w") as f:
    json.dump({"commit": COMMIT, "checkpoint": PRETRAINED, "chunks": len(info_paths), **summary}, f, indent=2)
"""),
    md("## 6 · Watch a rollout"),
    code("""
from IPython.display import Video
videos = sorted(glob.glob(f"{EVAL_ROOT}/**/*.mp4", recursive=True))
print(len(videos), "videos")
Video(videos[0], embed=True, width=480) if videos else None
"""),
]

# ---------------------------------------------------------------- notebook 02
nb02 = [
    md(f"""
# 02 · Train ACT from scratch (Step 1c)

{badge("02_train_act")}

Train our own ACT policy on `lerobot/aloha_sim_transfer_cube_human` (50 human demos) and compare it with the
pretrained checkpoint from notebook 01 (83 %). This run is also **the baseline all later encoder experiments
are compared against**.

Colab Free sessions die after a few hours, so training is **resumable**:
we train on the fast local disk and a background thread mirrors the newest checkpoints to Drive.
If the session dies, just reopen this notebook and run it top to bottom again: it continues from the last checkpoint on Drive.

| Section | When to run |
|---|---|
| 0–2 Setup + config | every session |
| 3 Timing test | once, before the first real run |
| 4 Train | every session until finished |
| 5–6 Evaluate + loss curve | when training is done |

**Runtime → Change runtime type → T4 GPU** before running.
"""),
    *SETUP,
    md("""
## 2 · Run configuration
Defaults follow the reference checkpoint's `train_config.json` (batch 8, lr 1e-5, seed 1000, no augmentation, 100k steps).
Change `RUN_NAME` for every new experiment: it is the folder name on Drive.
"""),
    code("""
import json, time
from avr.colab import run_streaming
from avr.background import start_background, watch
from avr.eval.chunked import run_chunked_eval
from avr.lerobot_cli import train_cmd, resume_cmd
from avr.checkpoints import CheckpointSyncer, restore_from_drive
from avr.train_log import read_train_log, seconds_per_step

RUN_NAME   = "act_transfer_cube_human_resnet18_s1000"
DATASET    = "lerobot/aloha_sim_transfer_cube_human"
TASK       = "AlohaTransferCube-v0"
STEPS      = 100_000  # as the reference checkpoint's train_config.json (model card says 80k)
BATCH_SIZE = 8
SAVE_FREQ  = 5_000    # adjust after the timing test: aim for a checkpoint every ~20-30 min
SEED       = 1000
USE_AMP    = False    # fp16 would be faster on a T4 but deviates from the reference recipe

LOCAL_RUN = f"/content/outputs/{RUN_NAME}"
DRIVE_RUN = f"{DRIVE_ROOT}/runs/{RUN_NAME}"
os.makedirs(DRIVE_RUN, exist_ok=True)
print("local:", LOCAL_RUN, "\\ndrive:", DRIVE_RUN)
"""),
    md("""
## 3 · Timing test (run once)
500 training steps without checkpoints or rollouts. Measures seconds per step on this GPU and whether the GPU
(`updt_s`) or video decoding / data loading (`data_s`) is the bottleneck. The first call also downloads the dataset.
"""),
    code("""
stamp = time.strftime("%Y%m%d-%H%M%S")
timing_dir = f"/content/outputs/timing_{stamp}"
timing_log = f"{DRIVE_ROOT}/runs/_timing/timing_{COMMIT}_{stamp}.log"

run_streaming(
    train_cmd(timing_dir, dataset=DATASET, task=TASK, steps=500, batch_size=BATCH_SIZE,
              log_freq=50, env_eval_freq=0, save_checkpoint=False, seed=SEED, use_amp=USE_AMP),
    log_path=timing_log,
)

recs = read_train_log(timing_log)
sps = seconds_per_step(recs)
updt = sum(r["updt_s"] for r in recs[1:]) / len(recs[1:])
data = sum(r["data_s"] for r in recs[1:]) / len(recs[1:])
print(f"\\n{sps:.3f} s/step  (max per log window: update {updt:.3f} s, data {data:.3f} s)")
print(f"{STEPS} steps ≈ {STEPS * sps / 3600:.1f} h of training (+ in-training rollouts)")
print(f"checkpoint every ~25 min ≈ SAVE_FREQ = {max(1000, round(25 * 60 / sps, -3)):.0f}")
if data > 0.5 * updt:
    print("⚠ data loading is a large share -> video decoding on 2 CPU cores is limiting")
"""),
    md("""
## 4 · Train (resumable)
Starts a fresh run, or resumes from the newest checkpoint on Drive if one exists.
Raising `STEPS` and re-running this cell extends a finished run from its last checkpoint.
Runs as a background job: the cell shows a status line every 2 min, the full log goes to `train.log` on Drive. Evaluation rollouts (10 episodes) run every 20k steps.
"""),
    code("""
config_path = restore_from_drive(DRIVE_RUN, LOCAL_RUN)
if config_path is not None:
    print("resuming from", os.path.realpath(config_path.parent.parent))
    cmd = resume_cmd(str(config_path), steps=STEPS)  # STEPS > stored total extends a finished run
else:
    assert not os.path.exists(LOCAL_RUN), f"{LOCAL_RUN} exists without checkpoints; delete it or change RUN_NAME"
    cmd = train_cmd(LOCAL_RUN, dataset=DATASET, task=TASK, steps=STEPS, batch_size=BATCH_SIZE,
                    save_freq=SAVE_FREQ, seed=SEED, use_amp=USE_AMP)
    meta = dict(run=RUN_NAME, commit=COMMIT, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                dataset=DATASET, task=TASK, steps=STEPS, batch_size=BATCH_SIZE,
                save_freq=SAVE_FREQ, seed=SEED, use_amp=USE_AMP, cmd=cmd)
    with open(f"{DRIVE_RUN}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

# background job: only a status line every 2 min in this cell, full log in train.log on Drive
train_log = f"{DRIVE_RUN}/train.log"
with CheckpointSyncer(LOCAL_RUN, DRIVE_RUN, keep=2, interval=60):
    watch(start_background(cmd, train_log), train_log, interval=120)
print("training finished")
"""),
    md("""
## 5 · Evaluate the final checkpoint
Same protocol as notebook 01 so the numbers are directly comparable.
"""),
    code("""
import glob
from avr.eval.stats import merge_eval_infos, summarize_eval, format_summary

if not os.path.exists(f"{LOCAL_RUN}/checkpoints/last"):
    restore_from_drive(DRIVE_RUN, LOCAL_RUN)
policy_dir = os.path.realpath(f"{LOCAL_RUN}/checkpoints/last/pretrained_model")
step = os.path.basename(os.path.dirname(policy_dir))
print("evaluating step", step)

N_EPISODES = 500  # same protocol as notebook 01: chunks of 100, seeds 1000-1499, resumable
eval_root = f"{DRIVE_RUN}/eval/step{step}_n{N_EPISODES}_seed1000"
info_paths = run_chunked_eval(policy_dir, eval_root, n_episodes=N_EPISODES, chunk_size=100, seed=1000, task=TASK)

summary = summarize_eval(merge_eval_infos(info_paths))
print(format_summary(summary))
with open(f"{eval_root}/summary.json", "w") as f:
    json.dump({"commit": COMMIT, "run": RUN_NAME, "step": int(step), **summary}, f, indent=2)
"""),
    md("""
### Paired comparison with the pretrained checkpoint (notebook 01)
Both were evaluated on the same 500 seeds, so we can compare episode by episode (exact McNemar test).
"""),
    code("""
from avr.eval.chunked import chunk_info_paths
from avr.eval.stats import mcnemar

def successes(root):
    return [bool(e["success"]) for e in merge_eval_infos(chunk_info_paths(root))["per_episode"]]

ref = successes(f"{DRIVE_ROOT}/eval/pretrained_transfer_cube/n500_seed1000")
ours = successes(eval_root)
t = mcnemar(ref, ours)
print(f"pretrained: {sum(ref)}/{len(ref)}   ours (step {step}): {sum(ours)}/{len(ours)}")
print(f"only pretrained succeeds: {t['only_a']}   only ours succeeds: {t['only_b']}   p = {t['p_value']:.2g}")
"""),
    md("## 6 · Training curve and in-training evaluations"),
    code("""
import ast, re
# in-training rollouts (10 episodes each, every 20k steps): the "Suite overall aggregated" log lines
evals = [ast.literal_eval(m.group(1)) for m in re.finditer(r"Suite overall aggregated: (\{.*\})", open(f"{DRIVE_RUN}/train.log").read())]
for i, e in enumerate(evals, 1):
    print(f"eval {i} (step ~{20_000 * i}): {e['pc_success']:.0f}% of {e['n_episodes']} episodes")

recs = read_train_log(f"{DRIVE_RUN}/train.log")
steps = [r["step"] for r in recs]
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(steps, [r["loss"] for r in recs])
ax.set_yscale("log"); ax.set_xlabel("step (approx., LeRobot rounds to 1K)"); ax.set_ylabel("train loss (L1 + KL)")
ax.set_title(RUN_NAME); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{DRIVE_RUN}/loss_curve.png", dpi=150)
"""),
    md("""
## 7 · Diagnostics: our training config vs. the reference checkpoint
Lists every field of `train_config.json` that differs (paths and logging fields excluded).
"""),
    code("""
import urllib.request
REF_URL = "https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/raw/main/train_config.json"
reference = json.load(urllib.request.urlopen(REF_URL))
ours_cfg = json.load(open(sorted(glob.glob(f"{DRIVE_RUN}/checkpoints/*/pretrained_model/train_config.json"))[-1]))

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out

IGNORE = ("output_dir", "job_name", "wandb", "pretrained_path", "repo_id", "resume", "checkpoint_path")
a, b = flatten(reference), flatten(ours_cfg)
for key in sorted(set(a) | set(b)):
    if any(part in key for part in IGNORE):
        continue
    if a.get(key, "<missing>") != b.get(key, "<missing>"):
        print(f"{key:55s} reference={a.get(key, '<missing>')!r:30.30s}  ours={b.get(key, '<missing>')!r:.40s}")
"""),
]


# ---------------------------------------------------------------- notebook 03
nb03 = [
    md(f"""
# 03 · Robustness under visual perturbations (Step 2a/2b)

{badge("03_robustness")}

Evaluates a policy under the perturbation suite in [`src/avr/perturb`](../src/avr/perturb):
11 factors (lighting, table color, background, distractors, camera pose, pixel noise, blur, JPEG, cube color)
x 4 severity levels. The perturbations are **purely visual**: unit tests check that robot and cube
physics are bit-identical to the unperturbed scene, only the camera image changes.
Each episode's perturbation is seeded by the episode seed, so every policy sees exactly the same
perturbed episodes (paired comparison).

| Section | What |
|---|---|
| 2 | Preview grid of all factors and levels |
| 3 | Validate our rollout runner against `lerobot-eval` (same seeds, same episodes) |
| 4 | Full suite on a checkpoint (resumable, runs in the background) |
| 5 | Results table + robustness curves |

**Colab Free:** only one GPU runtime at a time - run this after training in notebook 02 has finished
(or stopped at a checkpoint; it resumes later).
"""),
    *SETUP,
    md("## 2 · Preview: one row per factor, one column per severity level (seed 0)"),
    code("""
from IPython.display import display
from avr.perturb.preview import preview_grid

ROB_BASE = f"{DRIVE_ROOT}/robustness"
os.makedirs(ROB_BASE, exist_ok=True)
grid = preview_grid(seed=0, scale=0.3)
grid.save(f"{ROB_BASE}/preview_seed0.png")
display(grid)
"""),
    md("""
## 3 · Validate the runner against `lerobot-eval`
Our runner steps `PerturbedAlohaEnv` itself (needed for the perturbations). With no perturbation and the same
seeds (1000-1099) it should reproduce the episodes of notebook 01's first chunk. Small differences are possible
(batch-of-5 vs. single-episode GPU numerics), large ones would mean a bug.
"""),
    code("""
import json
from avr.background import start_background, watch
from avr.eval.chunked import chunk_info_paths
from avr.eval.stats import merge_eval_infos

POLICY_DIR = f"{DRIVE_ROOT}/checkpoints/act_aloha_sim_transfer_cube_human_migrated"
ROB_ROOT = f"{ROB_BASE}/pretrained_resnet18"

def run_suite(policy, out, factors, levels=(1, 2, 3, 4), episodes=50, seed=1000, tag="suite"):
    cmd = [sys.executable, "-m", "avr.eval.robustness", "--policy", policy, "--out", out,
           "--factors", *factors, "--levels", *map(str, levels),
           "--episodes", str(episodes), "--seed", str(seed)]
    log = f"/content/logs/{os.path.basename(out)}_{tag}.log"
    watch(start_background(cmd, log), log, interval=60)

run_suite(POLICY_DIR, ROB_ROOT, ["clean"], episodes=100, tag="clean")

ours = json.load(open(f"{ROB_ROOT}/clean/level0.json"))["episodes"]
ref = merge_eval_infos(chunk_info_paths(f"{DRIVE_ROOT}/eval/pretrained_transfer_cube/n500_seed1000")[:1])["per_episode"]
agree = sum(a["success"] == b["success"] for a, b in zip(ours, ref))
print(f"runner: {sum(e['success'] for e in ours)}/{len(ours)}   lerobot-eval: {sum(e['success'] for e in ref)}/{len(ref)}")
print(f"same outcome in {agree}/{len(ours)} paired episodes")
"""),
    md("""
## 4 · Full suite (11 factors x 4 levels x 50 episodes)
Seeds 1000-1049 for every cell. Finished cells are stored on Drive as `<factor>/level<k>.json` and skipped
on re-run, so the suite can be spread over several sessions. Two videos per cell are saved to `videos/`.
The status line shows the running success rate of the current cell.
"""),
    code("""
from avr.perturb import FACTORS
run_suite(POLICY_DIR, ROB_ROOT, sorted(FACTORS), episodes=50)
"""),
    md("## 5 · Results"),
    code("""
import pandas as pd
from avr.analysis.robustness import load_cells, plot_robustness, summary_rows

cells = load_cells(ROB_ROOT)
clean_rate = cells[("clean", 0)]["episodes"]
clean_rate = sum(e["success"] for e in clean_rate) / len(clean_rate)

df = pd.DataFrame(summary_rows(cells))
df.to_csv(f"{ROB_ROOT}/summary.csv", index=False)
pct = {c: "{:.0%}" for c in ["success", "ci_lo", "ci_hi", "fail_no_touch", "fail_grasp", "fail_transport"]}
display(df.style.format(pct).hide(axis="index"))

label = "pretrained ACT (ResNet18)"
fig = plot_robustness({label: cells}, {label: clean_rate}, path=f"{ROB_ROOT}/robustness_curves.png")
"""),
    md("## 6 · Failure videos"),
    code("""
import glob
from IPython.display import Video
fails = sorted(glob.glob(f"{ROB_ROOT}/videos/*_fail.mp4"))
print(len(fails), "failure videos, e.g.:", *[os.path.basename(f) for f in fails[:10]], sep="\\n  ")
Video(fails[0], embed=True, width=480) if fails else None
"""),
]

for name, cells in (("01_eval_pretrained", nb01), ("02_train_act", nb02), ("03_robustness", nb03)):
    with open(NB_DIR / f"{name}.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook(cells), f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", name)
