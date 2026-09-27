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

# installs lerobot[aloha] pinned in pyproject.toml (takes a few minutes) and the act_enc policy plugin
!pip install -q -e ".[sim]" -e plugins/lerobot_policy_act_enc
sys.path.insert(0, f"{REPO_DIR}/src")  # editable install is only picked up after a restart
# re-running this cell in a live session must pick up the freshly pulled code:
# drop already imported avr modules so the next import loads the new version
for name in [m for m in sys.modules if m == "avr" or m.startswith("avr.")]:
    del sys.modules[name]

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
# 02 · Train ACT: baseline (Step 1c) and encoder variants (Step 2c)

{badge("02_train_act")}

Train ACT on `lerobot/aloha_sim_transfer_cube_human` (50 human demos) with the reference recipe.
`VARIANT = "baseline"` is plain LeRobot ACT (our baseline, 77.4 % at 100k steps); any other value trains ACT with a
different visual encoder through the `act_enc` plugin (`plugins/lerobot_policy_act_enc`), everything else unchanged.
Evaluation (section 5) is identical for all variants and paired with the pretrained checkpoint on the same seeds.

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

# What to train: "baseline" = plain LeRobot ACT (ImageNet ResNet18, finetuned), or an encoder variant
# "<encoder>_<frozen|ft>" via the act_enc plugin, e.g. "dinov2_vits14_frozen", "dinov2_vits14_ft",
# "resnet18_scratch_ft", "clip_vitb16_frozen", "siglip_vitb16_frozen" (encoders: src/avr/encoders.py)
VARIANT    = "baseline"

if VARIANT == "baseline":
    RUN_NAME, POLICY_TYPE, POLICY_ARGS = "act_transfer_cube_human_resnet18_s1000", "act", {}
else:
    encoder, mode = VARIANT.rsplit("_", 1)
    assert mode in ("frozen", "ft"), VARIANT
    RUN_NAME = f"actenc_transfer_cube_human_{VARIANT}_s1000"
    POLICY_TYPE, POLICY_ARGS = "act_enc", {"encoder": encoder, "freeze_encoder": mode == "frozen"}
print("run:", RUN_NAME, "| policy:", POLICY_TYPE, POLICY_ARGS)
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
    train_cmd(timing_dir, policy_type=POLICY_TYPE, policy_args=POLICY_ARGS, dataset=DATASET, task=TASK,
              steps=500, batch_size=BATCH_SIZE, log_freq=50, env_eval_freq=0, save_checkpoint=False,
              seed=SEED, use_amp=USE_AMP),
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
    cmd = train_cmd(LOCAL_RUN, policy_type=POLICY_TYPE, policy_args=POLICY_ARGS, dataset=DATASET, task=TASK,
                    steps=STEPS, batch_size=BATCH_SIZE, save_freq=SAVE_FREQ, seed=SEED, use_amp=USE_AMP)
    meta = dict(run=RUN_NAME, variant=VARIANT, commit=COMMIT, started=time.strftime("%Y-%m-%d %H:%M:%S"),
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

`POLICIES` (from `src/avr/registry.py`) lists every checkpoint we compare; pick the one to run with
`SUITE_POLICY`. Encoder variants trained in notebook 02 appear automatically. Our own 100k run is the baseline for the encoder comparison,
since all variants are trained with exactly its recipe.
"""),
    code("""
from avr.perturb import FACTORS

from avr.registry import policy_registry

# every policy we compare (encoder variants trained in notebook 02 are discovered automatically)
POLICIES = policy_registry(DRIVE_ROOT)
print(*POLICIES, sep="\\n")
SUITE_POLICY = "ours, ResNet18 ImageNet (100k)"

policy_path, SUITE_ROOT = POLICIES[SUITE_POLICY].checkpoint, POLICIES[SUITE_POLICY].robustness
assert os.path.exists(f"{policy_path}/config.json"), policy_path
# clean run on seeds 1000-1099: the paired reference for every perturbed cell (seeds 1000-1049)
run_suite(policy_path, SUITE_ROOT, ["clean"], episodes=100, tag="clean")
MAIN_FACTORS = [f for f in sorted(FACTORS) if FACTORS[f].category != "diagnostic"]
run_suite(policy_path, SUITE_ROOT, MAIN_FACTORS, episodes=50)
"""),
    md("""
### 4b · Shadow diagnostics
`light_direction` hurts far more than light color or intensity. Hypothesis: from the top camera, height is
hard to see, and the policy uses the **shadows** of grippers and cube as a height cue. Two diagnostic factors:
- `shadows_off` - no shadows, lights unchanged (only level 1 is meaningful)
- `light_direction_noshadow` - exactly the same light rotation per episode as `light_direction`, but without shadows

If shadows are the cue, `shadows_off` should hurt, and removing the shadows should *not* rescue the rotated-light episodes
the way it would if shading alone were the problem.
"""),
    code("""
run_suite(policy_path, SUITE_ROOT, ["shadows_off"], levels=[1], episodes=50, tag="shadows")
run_suite(policy_path, SUITE_ROOT, ["light_direction_noshadow"], episodes=50, tag="noshadow")
"""),
    md("## 5 · Results"),
    code("""
import pandas as pd
from avr.analysis.robustness import load_cells, plot_robustness, summary_rows

# every policy with suite results; level 0 = its clean success rate from the 500-episode lerobot-eval
results, clean_rates = {}, {}
for label, (_, root, clean_eval, _) in POLICIES.items():
    cells = {k: v for k, v in load_cells(root).items() if k[0] != "clean"}
    if not cells:
        continue
    results[label] = cells
    eps = merge_eval_infos(chunk_info_paths(clean_eval))["per_episode"]
    clean_rates[label] = sum(bool(e["success"]) for e in eps) / len(eps)
print({k: f"{len(v)} cells, clean {clean_rates[k]:.1%}" for k, v in results.items()})

# paired against the clean run of the same policy on the same seeds; p_holm corrects for all cells
clean_cell = load_cells(SUITE_ROOT)[("clean", 0)]
df = pd.DataFrame(summary_rows(results[SUITE_POLICY], clean=clean_cell))
df.to_csv(f"{SUITE_ROOT}/summary.csv", index=False)
cols = ["factor", "level", "category", "n", "success", "ci_lo", "ci_hi", "clean_same_seeds", "delta",
        "only_clean", "only_perturbed", "p_holm", "fail_no_touch", "fail_grasp", "fail_transport"]
fmt = {c: "{:.0%}" for c in ["success", "ci_lo", "ci_hi", "clean_same_seeds", "fail_no_touch", "fail_grasp", "fail_transport"]}
fmt.update(delta="{:+.0%}", p_holm="{:.2g}")
display(df[cols].style.format(fmt).hide(axis="index"))

fig = plot_robustness(results, clean_rates, path=f"{ROB_BASE}/robustness_curves.png")
"""),
    md("""
### 5b · Shadow hypothesis
Per level: success with the rotated light, with the same rotation but no shadows, and the paired test between the two
(same seeds, same light rotation). Plus `shadows_off` against the clean run.
"""),
    code("""
from avr.eval.stats import mcnemar

cells_all = load_cells(SUITE_ROOT)
def succ(cell):
    return {e["seed"]: e["success"] for e in cell["episodes"]}

rows = []
for level in range(1, 5):
    a, b = succ(cells_all[("light_direction", level)]), succ(cells_all[("light_direction_noshadow", level)])
    seeds = sorted(a)
    t = mcnemar([a[x] for x in seeds], [b[x] for x in seeds])
    rows.append({"level": level, "rotated light": sum(a.values()) / len(a),
                 "rotated light, no shadows": sum(b.values()) / len(b),
                 "only with shadows": t["only_a"], "only without": t["only_b"], "p": t["p_value"]})
display(pd.DataFrame(rows).style.format({"rotated light": "{:.0%}", "rotated light, no shadows": "{:.0%}", "p": "{:.2g}"}).hide(axis="index"))

c, so = succ(clean_cell), succ(cells_all[("shadows_off", 1)])
t = mcnemar([c[x] for x in sorted(so)], [so[x] for x in sorted(so)])
print(f"shadows_off: {sum(so.values())}/{len(so)} vs clean {sum(c[x] for x in so)}/{len(so)} on the same seeds "
      f"(only clean {t['only_a']}, only shadows_off {t['only_b']}, p = {t['p_value']:.2g})")
"""),
    md("## 6 · Failure videos"),
    code("""
import glob
from IPython.display import Video
def videos(pattern):
    return sorted(glob.glob(f"{SUITE_ROOT}/videos/{pattern}"))

# a light_direction failure and a successful recolored cube (cube_color level 4)
for path in videos("light_direction_level1_*_fail.mp4")[:1] + videos("cube_color_level4_*_ok.mp4")[:1]:
    print(os.path.basename(path))
    display(Video(path, embed=True, width=480))
"""),
]


# ---------------------------------------------------------------- notebook 04
nb04 = [
    md(f"""
# 04 · Why does a perturbation hurt? Probing the policy (Step 2d)

{badge("04_probe")}

Every perturbation is purely visual, so a clean episode's actions can be **replayed** under any perturbation and the
robot passes through exactly the same states - only the camera image differs. At 8 probe steps per episode we feed
the clean and the perturbed image (same state) through the policy and measure:

| Metric | Meaning |
|---|---|
| `feature_shift` | cosine distance between the encoder's feature maps - how much the *representation* moves |
| `action_shift` | mean abs. difference of the predicted 100-step action chunks (rad) - how much the *decision* moves |
| `attn_shift` | total variation between the decoder's cross-attention maps |

Plus, for the first seed, **occlusion sensitivity** maps (how much the action chunk changes when a gray patch hides each
image location), with the shadows outlined. No closed-loop rollouts under perturbation are needed: ~10 min per policy.
"""),
    *SETUP,
    md("## 2 · Run the probe for every trained policy (resumable per seed)"),
    code("""
from avr.background import start_background, watch
from avr.registry import policy_registry, trained

POLICIES = trained(policy_registry(DRIVE_ROOT))
print(*POLICIES, sep="\\n")
N_SEEDS = 10

for label, paths in POLICIES.items():
    print(f"\\n=== {label}")
    cmd = [sys.executable, "-m", "avr.analysis.features", "--policy", paths.checkpoint,
           "--out", paths.probe, "--seeds", str(N_SEEDS), "--seed", "1000"]
    log = f"/content/logs/probe_{os.path.basename(paths.probe)}.log"
    watch(start_background(cmd, log), log, interval=60)
"""),
    md("""
## 3 · Probe metrics per perturbation
Mean over 10 seeds (each seed averaged over its probe steps first), per policy.
"""),
    code("""
import pandas as pd
from IPython.display import display
from avr.analysis.attention import load_probe_rows, shift_table

probe_rows = {label: load_probe_rows(f"{p.probe}/feature_probe.jsonl") for label, p in POLICIES.items()
              if os.path.exists(f"{p.probe}/feature_probe.jsonl")}
for label, rows in probe_rows.items():
    print(label)
    df = pd.DataFrame(shift_table(rows))[["factor", "level", "feature_shift", "action_shift", "attn_shift"]]
    display(df.style.format({"feature_shift": "{:.3f}", "action_shift": "{:.4f}", "attn_shift": "{:.3f}"}).hide(axis="index"))
"""),
    md("""
## 4 · Does the probe predict the damage?
One point per factor x level: probe shift (x) against the paired success change from the perturbation suite
(notebook 03, y). A high rank correlation means the cheap probe explains where the policy fails.
Only policies with suite results are shown.
"""),
    code("""
from avr.analysis.attention import plot_shift_vs_drop
from avr.analysis.robustness import load_cells, summary_rows

pairs = {}
for label, rows in probe_rows.items():
    cells = load_cells(POLICIES[label].robustness)
    if ("clean", 0) not in cells:
        continue
    suite = summary_rows({k: v for k, v in cells.items() if k[0] != "clean"}, clean=cells[("clean", 0)])
    pairs[label] = (rows, suite)

PROBE_FIG = f"{DRIVE_ROOT}/probe"
for metric in ("action_shift", "feature_shift"):
    fig, rhos = plot_shift_vs_drop(pairs, metric=metric, path=f"{PROBE_FIG}/{metric}_vs_drop.png")
    print(metric, {k: round(v, 2) for k, v in rhos.items()})
"""),
    md("""
## 5 · Where does the policy look?
Occlusion sensitivity (bottom) for the clean image and five perturbations, first seed, probe steps 50 and 100.
Orange outlines: pixels that change when shadows are switched off, i.e. the shadows.
"""),
    code("""
import glob
from avr.analysis.attention import attention_figure, load_figure_maps

for label, paths in POLICIES.items():
    figs = sorted(glob.glob(f"{paths.probe}/figures_seed*.npz"))
    if not figs:
        continue
    maps = load_figure_maps(figs[0])
    for step in sorted(maps["None_L0"]):
        attention_figure(maps, step, kind="occl", title=f"{label}: occlusion sensitivity, step {step}",
                         path=f"{paths.probe}/occlusion_step{step}.png")
"""),
    md("### 5b · For comparison: the decoder cross-attention (diffuse, piles up on empty background)"),
    code("""
for label, paths in POLICIES.items():
    figs = sorted(glob.glob(f"{paths.probe}/figures_seed*.npz"))
    if figs:
        maps = load_figure_maps(figs[0])
        attention_figure(maps, sorted(maps["None_L0"])[0], kind="attn", title=f"{label}: cross-attention")
"""),
]

for name, cells in (("01_eval_pretrained", nb01), ("02_train_act", nb02), ("03_robustness", nb03), ("04_probe", nb04)):
    with open(NB_DIR / f"{name}.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook(cells), f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote", name)
