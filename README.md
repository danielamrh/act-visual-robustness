# How Robust Are Visuomotor Imitation Policies to Visual Shift?

Self-study final project following Stanford
[CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/) (Spring 2026 curriculum).

Imitation-learning policies like [ACT](https://tonyzhaozh.github.io/aloha/) see the world only through
a camera, and small visual changes (different lighting, a new table colour, a distractor object, a
slightly moved camera) can break them. This project measures **which visual representations and
augmentations make an ACT policy robust**, and why, in the ALOHA simulation.

## Plan

| Step | Goal | Status |
|---|---|---|
| 1 · Baseline | Reproduce ACT on `AlohaTransferCube-v0` (pretrained checkpoint + own training) | 🚧 in progress |
| 2 · Analysis | Perturbation suite (lighting, textures, distractors, camera pose, image noise); compare visual encoders (ResNet18 scratch / ImageNet, DINOv2, CLIP, R3M; frozen vs finetuned); feature-shift and attention analysis | ⏳ |
| 3 · Method | Targeted modification derived from the analysis | ⏳ |

## Results

*Coming soon.*

## Quickstart (Google Colab)

| Notebook | What it does |
|---|---|
| [`01_eval_pretrained`](notebooks/01_eval_pretrained.ipynb) <a href="https://colab.research.google.com/github/danielamrh/act-visual-robustness/blob/main/notebooks/01_eval_pretrained.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> | Evaluate the pretrained LeRobot ACT checkpoint (success rate, 95% CI, failure stages) |
| [`02_train_act`](notebooks/02_train_act.ipynb) <a href="https://colab.research.google.com/github/danielamrh/act-visual-robustness/blob/main/notebooks/02_train_act.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a> | Train ACT from scratch, resumable across Colab sessions (checkpoints mirrored to Drive) |

Code lives in this repo, outputs (checkpoints, eval videos) go to Google Drive under
`MyDrive/act_robustness/`. Every run records the git commit it was produced with.

## Repository layout

```
configs/      one config per experiment
notebooks/    Colab entry points
results/      final tables, plots and GIFs
src/avr/      project code: CLI wrappers, checkpoint sync, log parsing, eval statistics
tests/        unit tests
```

## Local development

```bash
pip install -e ".[dev]"      # the sim extra (LeRobot) needs Python >= 3.12
nbstripout --install         # strip notebook outputs before committing
pytest
```

## References

- T. Z. Zhao, V. Kumar, S. Levine, C. Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT / ALOHA). RSS 2023.
- [LeRobot](https://github.com/huggingface/lerobot) (Apache 2.0): ACT implementation, datasets, training and evaluation.
- [gym-aloha](https://github.com/huggingface/gym-aloha): ALOHA simulation environments.

## Acknowledgements

Parts of the code were written with the help of an AI coding assistant (Claude Code); all experiments,
analysis and conclusions are my own.

## License

MIT
