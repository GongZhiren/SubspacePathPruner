# SubspacePath Pruner

**Inference-time, training-free structured pruning for LLMs via probe-based representation–parameter coupling.**

- **Paper:** [OpenReview (ICML 2026)](https://openreview.net/forum?id=fvkCjFvWKf)
- **Project Page:** [https://gongzhiren.github.io/SubspacePathPruner-website/](https://gongzhiren.github.io/SubspacePathPruner-website/)

SubspacePath Pruner compiles a *scenario-specific* pruned subnetwork of a frozen
large language model at inference time — **no fine-tuning, no gradient updates, no
scenario training data**. It exploits a simple observation: representation
subspaces that align with semantic *domains* in embedding space are coupled with
sparse, reusable *attention-head pathways* in parameter space. A handful of
lightweight linear probes, trained once offline, are enough to map a scenario's
domain mixture onto the heads that matter, and to prune the rest under a budget.

The released code lets you:

- **Run pruned inference out of the box** using the pretrained probe/importance/whitelist
  artifacts shipped for four models (Qwen2.5-7B, Qwen2.5-14B, Llama-3.1-8B, Llama-2-13B-chat).
- **Reproduce the offline stage** (probe training → calibration → head importance →
  whitelist) on your own model with the included input pools.

---

## Method

The framework has two components, both run **offline once**; the online stage is
purely a compilation step.

### DBS — Domain-Basis Synthesis (offline)

A compact set of quasi-orthogonal **domain axes** is constructed in embedding space
to serve as a stable coordinate system. In this release the domains are
pre-selected, so the axes reduce to a one-hot basis over the selected domains
(`src/preorientation/domain_axes.py`). You do **not** need to re-run domain
selection to use the code.

### PSP — Probe-based Scenario Pruning

**Offline (training-free for the model):**

1. **Layer-wise linear probes** — for each layer ℓ and domain *k*, a 1-vs-rest
   probe is trained on the post-attention residual stream to score domain
   relevance. Only the *probes* are trained; the base model is frozen.
   (`src/preorientation/linear_probe.py`)
2. **Temperature calibration** — probe logits are calibrated with temperature
   scaling for single-domain, OOD, and cross-domain regimes.
   (`src/preorientation/probe_calibration.py`)
3. **Axis-aligned head importance** `I_{ℓ,h,k}` — the expected squared projection
   of each head's residual write onto domain axis *k*:

   ```
   I_{ℓ,h,k} = E_{x ~ P_k} [ (u_k^T w̃_{ℓ,h}(x))^2 / (||w̃_{ℓ,h}(x)||^2 + ε) ]
   ```
   (`src/probe/head_importance.py`)
4. **Whitelist** — domain-invariant "backbone" heads (low importance variance
   across domains, high mean importance, confirmed by a statistical test) that are
   always kept. (`src/probe/whitelist_identification.py`)

**Online (per scenario, training-free):**

1. **Diagnose** the scenario's domain mixture by evaluating the probes on the first
   turn(s); compute a normalized-entropy **scenario breadth** `c(s) ∈ [0,1]`.
   (`src/probe/domain_inference.py`)
2. **Score** each head by combining its calibrated domain relevance with the cached
   importance: `score_{ℓ,h}(s) = Σ_k s_k^eff · I_{ℓ,h,k}`.
3. **Compile** a binary head-pruning mask under a budget governed by
   `pruning_strength` (higher prunes more), always keeping the whitelist, and
   **reuse the same mask for every turn** of the scenario — zero per-turn overhead.
   (`src/probe/session_pruning.py`)

Because nothing is optimized online, scenario compilation takes tens of
milliseconds and the pruned model is just a structured head mask over the
original frozen weights.

---

## Repository layout

```
subspacepath-pruner/
├── src/
│   ├── model/            # BaseModel wrapper (load, quantize, head masking)
│   ├── preorientation/   # DBS axes, linear probes, temperature calibration, foundation layers
│   ├── probe/            # head importance, whitelist, domain inference, session pruning
│   ├── reasoning/        # single-step inference + official chat templates
│   ├── evaluation/       # exact-match answer scoring + result logging
│   └── utils/            # logging, config, device helpers
├── scripts/
│   ├── run_inference.py  # ONLINE: prune + infer + evaluate (history OFF by default)
│   └── train_offline.py  # OFFLINE: train probes → calibrate → importance → whitelist
├── data/
│   ├── train/  val/      # per-domain input pools (input-only, no labels)
│   └── test/             # evaluation scenarios (see "Data" below)
├── outputs/<model>/ppd_pipeline/   # pretrained artifacts for 4 models (see below)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone <your-fork-url> subspacepath-pruner
cd subspacepath-pruner
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch ≥ 2.0, Transformers ≥ 4.40, on a single CUDA GPU.

**Models are not bundled.** Download the HuggingFace weights you want and place them
under `models/`, e.g.:

```
models/Qwen2.5-7B-Instruct/      # standard HF model directory
models/llama-3.1-8b/
models/Qwen14B/                  # Qwen2.5-14B-Instruct
models/Llama-2-13b-chat-hf/
```

---

## Quick start — pruned inference (uses shipped artifacts)

The offline artifacts for the four models are already in
`outputs/<model>/ppd_pipeline/`, so you can run pruned inference immediately:

```bash
python scripts/run_inference.py \
    --model Qwen2.5-7B-Instruct \
    --gpu 0 \
    --pruning_strengths 0.2 0.4 0.6 \
    --num_samples 20
```

This loads the probes, head importance and whitelist, sweeps the requested pruning
strengths over a 20-scenario sample from each test category, and prints a summary
of accuracy vs. average pruned-head fraction. Per-run results are written to
`outputs/<model>/pruning_strength/`.

Useful flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `Qwen2.5-7B-Instruct` | directory name under `models/` |
| `--pruning_strengths` | `0.2 0.4 0.6` | strengths to sweep (higher → more pruning) |
| `--datasets` | `selected_domain out_of_domain cross_domain` | which `data/test/*` categories to run |
| `--num_samples` | `20` | scenarios sampled per category (`-1` = all) |
| `--use_history` | `false` | include previous turns as context |
| `--use_calibration` | `false` | use the temperature-calibrated probe system |

To also evaluate the cross-dataset splits, add e.g.
`--datasets selected_domain out_of_domain cross_domain commonsenseqa natural_questions arc`.

### Reproducing the paper results

The defaults above (`--num_samples 20`, three strengths) are tuned for a quick
first run. To reproduce the full experiment, evaluate the **entire** test set over
the full pruning-strength sweep:

```bash
python scripts/run_inference.py \
    --model Qwen2.5-7B-Instruct --gpu 0 \
    --datasets selected_domain out_of_domain cross_domain \
    --num_samples -1 \
    --pruning_strengths 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
```

`--num_samples -1` uses every scenario in each category (no sub-sampling). Repeat
with the cross-dataset categories (`commonsenseqa natural_questions arc`) for the
cross-dataset table. Swap `--model` for `Qwen14B`, `llama-3.1-8b`, or
`Llama-2-13b-chat-hf` to cover the other models. Accuracy is reported as the
exact-match score described under *Evaluation protocol*.

---

## Reproduce the offline stage

To regenerate the artifacts for a model from the included input pools:

```bash
python scripts/train_offline.py \
    --model Qwen2.5-7B-Instruct \
    --gpu 0 \
    --selected_domains chemistry finance history math philosophy technology \
    --final_probe_epochs 20
```

Steps performed (base model stays **frozen** throughout):

1. train the layer-wise linear probes,
2. temperature-calibrate them (4-probe system),
3. compute axis-aligned head importance `I_{ℓ,h,k}`,
4. identify the domain-invariant head whitelist.

Outputs are written to `outputs/<model>/ppd_pipeline/`:
`probe1_base.pt`, `probe_temperatures.json`, `calibration/`, `head_importance.pt`,
`whitelist.json` — exactly the files `run_inference.py` consumes.

`--head_importance single` (default) computes one importance set from the base
probe and is fast; `--head_importance multi` computes scenario-specific
(single/OOD/cross-domain) sets and uses the cross-domain one.

> The default `--selected_domains` order matches the per-domain files in
> `data/train/`. The online pruner is **index-based** and internally consistent
> with whatever order you train on, so domain *names* never affect inference — they
> only label the probe indices.

---

## Data

All data ships with the repository.

- **`data/train/`, `data/val/`** — input-only pools, one JSON per selected domain
  (`chemistry, finance, history, math, philosophy, technology`). Each file is a
  list of raw input strings; no labels are needed to train the probes.
- **`data/test/`** — multi-turn evaluation scenarios. Each scenario has a
  `topic_description` and a list of `turns`, where every turn has
  `prompt`, `answer`, and `task_type` (`multiple_choice` / `factual` / `code` / `reasoning`).
  Categories:
  - `selected_domain/` — in-distribution domains,
  - `out_of_domain/` — held-out domains (OOD),
  - `cross_domain/` — multi-domain mixtures,
  - `commonsenseqa/`, `natural_questions/`, `arc/` — cross-dataset splits.

---

## Pretrained artifacts

`outputs/<model>/ppd_pipeline/` ships, for each of the four models:

| File | Contents |
|------|----------|
| `probe1_base.pt` | trained layer-wise linear probes (the base probe) |
| `final_probes.pt` | probes from the final training pass |
| `probe_temperatures.json` | per-regime / per-layer temperature-scaling parameters |
| `calibration/` | cross-dataset calibration temperatures |
| `head_importance.pt` | axis-aligned head importance `I_{ℓ,h,k}` |
| `whitelist.json` | list of always-kept `(layer, head)` pairs |

These are the only learned parameters in the method (a few MB per model). The base
model weights themselves are never modified.

---

## Evaluation protocol

Answer accuracy is **exact-match (EM) keyword overlap**: after stopword removal,
the fraction of the expected answer's keywords matched by the prediction (for
multiple-choice, the extracted option letter must match exactly). There is **no
semantic-similarity scoring and no LLM judge** — see `src/evaluation/answer_evaluator.py`.

---

## Notes

- Single-GPU by design; select the device with `--gpu`. For large models, load with
  `--quantization int4` in `train_offline.py`.
- History (multi-turn conversation context) is **off by default**; enable with
  `--use_history true`.
- `peft` is only needed if your base model path is a LoRA/PEFT adapter; it is an
  optional dependency.

## License

Released under the MIT License — see [LICENSE](LICENSE).
