# Looped-MoE: training, evaluation and probing code

*Anonymized for double-blind review.*

`LoopMoE` is this repository's internal code name for the looped MoE models studied in the paper (Foil and its baselines). It is unrelated to the LoopMoE method proposed by Chen et al. (2026, arXiv:2606.04438). A later version will rename it to Foil in the code.

## Model weights

The weights of the models in the paper are available: the final checkpoints of
the four 100B-token models (Foil-1, Foil-2, Foil-3 and Base) and the intermediate
checkpoints of each run. To preserve anonymity during review, they are not
linked here; they will be released publicly after the review period. Reviewers
who would like the weights now can ask through OpenReview, and we will share
them through an anonymous link.

## Code

Code for pretraining the looped mixture-of-experts language models in the paper,
evaluating them, and running the routing probe and expert-masking experiments.
All commands are run from the repository root: code under `src/` as modules
(`python -m src....`, the code uses absolute `src.` imports), the standalone data
scripts under `scripts/data/` directly (`python scripts/data/<name>.py`).

**Running outside a git checkout** (for example from a downloaded zip): the
tokenization, training, evaluation, probe and masking entry points record the
exact code revision they ran, read from git, and refuse to run without it. Set
the environment variable `CODE_REVISION_JSON` to the path of a JSON file with the
fields `commit` (full 40-character SHA), `dirty`, `diff_sha256` and `describe`
(checked in `src/metrics/provenance.py`). For example, for a zip of this
repository, run from the repository root before creating any file inside it:

```bash
TREE_SHA=$(find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
cat > code_revision.json <<JSON
{"commit": "<40-character SHA of the commit the zip was made from>",
 "dirty": true,
 "diff_sha256": "$TREE_SHA",
 "describe": "zip download, not a git checkout"}
JSON
export CODE_REVISION_JSON=$PWD/code_revision.json
```

`diff_sha256` here is a hash over every file in the tree, so any local change
shows up in the recorded provenance.

The looped model code (`src/model/modeling_loop_lm.py`) is ported from the
release accompanying *Sparse Layers are Critical to Scaling Looped Language
Models* (arXiv:2605.09165, Hugging Face repos
`ml-ryanlee/seedvar-looped-moe-1e18-d704-seed42..47`, Apache-2.0). See `NOTICE`.

## Installation

Three separate environments, one requirements file each (Python 3.11). Training
and evaluation cannot share an environment: they pin different versions of
torch (2.12.0 vs 2.13.0) and datasets (3.6.0 vs 2.21.0). The training and
evaluation sequences below were verified in fresh virtual environments.

In a fresh environment, upgrade pip first: older bundled pip versions (22.3.1,
for example) reject a wheel on the PyTorch index and the torch install fails.

```bash
# Training (tokenization and pretraining)
python3.11 -m venv .venv-train && . .venv-train/bin/activate
python -m pip install pip==26.2.1
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
pip install --no-deps "torchtitan @ git+https://github.com/pytorch/torchtitan.git@73a0e6979dd10b6b1904098eb3c8f62c18ab87ce" megatron-core==0.16.1
grep -vE '^(torchtitan|megatron-core)[ =@]' requirements-train.txt | pip install -r /dev/stdin

# Evaluation, held-out loss, held-out slice building, probing, expert masking
python3.11 -m venv .venv-eval && . .venv-eval/bin/activate
python -m pip install pip==26.2.1
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install --no-deps megatron-core==0.16.1
grep -vE '^megatron-core[ =]' requirements-eval.txt | pip install -r /dev/stdin
export HF_DATASETS_TRUST_REMOTE_CODE=1

# Fine-tuning probe
python3.11 -m venv .venv-posttrain && . .venv-posttrain/bin/activate
python -m pip install pip==26.2.1
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-posttrain.txt
```

The `cu126` wheel index is for GPU drivers that support CUDA 12.x but not 13;
otherwise the default torch wheel works (our evaluation environment used the
cu130 build; cu126 was also verified end to end). `megatron-core` is installed
without its dependencies; importing it prints a harmless warning that
Transformer Engine and Apex are not installed. `wandb` is installed with the
training requirements and used only when training runs with
`--metrics.enable_wandb`.

## Data preparation

Pretraining uses FineWeb-Edu `sample-100BT` (140 parquet files, about 286 GB),
tokenized with the SmolLM2 tokenizer (vocabulary 49,152). The SmolLM2 tokenizer is
bundled under `src/data/artifacts/tokenizer/smollm2/`, so tokenization needs no
download.

1. **Download the raw shards** (training environment). Resumable: rerun the same command after an
   interruption and finished shards are skipped. The script writes into
   `<out-dir>/<config>/`, so the command below puts the 140 parquet files and
   their `raw_manifest.json` in `pretrain/data/raw/sample-100BT/`. The shard names
   and sha256 used for the paper are in `src/data/artifacts/fw100bt_raw_manifest.json`.
   ```bash
   python -m src.data.download_shards --config sample-100BT --num-shards 140 \
       --out-dir pretrain/data/raw/
   ```
2. **Tokenize into Megatron bin/idx shards** (training environment; one shard per
   parquet file).
   `--raw-dir` is the directory step 1 wrote. The output directory below,
   `pretrain/data/tokenized/sample-100BT`, is the default that the training
   command generator reads (pass `--data` to it if you use another one).
   ```bash
   python -m src.data.tokenize_to_bin_idx --raw-dir pretrain/data/raw/sample-100BT \
       --out-dir pretrain/data/tokenized/sample-100BT --tokenizer-name smollm2 \
       --num-workers <N>
   ```
3. **Build the held-out validation slices** (evaluation environment; never seen in
   pretraining; each script proves non-overlap with the training documents). Needs network access;
   all sources are public datasets. The non-overlap check downloads all 140
   training shards first (about 286 GB in total; each is deleted once its
   document ids and URLs are read). The domain builder needs the same training
   ids and URLs: its first run builds them the same way and caches them in
   `_runs/eval/heldout/_train_keys/`, which later domains reuse. Building the
   FineWeb-Edu slice peaks at about 15 GB of memory. Outside a git checkout, each
   slice's `manifest.json` records `generating_commit` as `"unknown"`. These are the
   exact
   commands that produced the slices used in the paper (FineWeb-Edu 8,000,545
   tokens; the five domain slices 3.0–4.0M tokens each).
   ```bash
   python -m src.data.build_fineweb_edu_heldout \
       --out-dir _runs/eval/heldout/fineweb_edu_heldout_v1 \
       --target-tokens 8000000 --tokenizer-name smollm2
   python -m src.data.build_domain_heldout --domain wikipedia --target-tokens 4000000 --tokenizer-name smollm2
   python -m src.data.build_domain_heldout --domain arxiv     --target-tokens 3000000 --tokenizer-name smollm2
   python -m src.data.build_domain_heldout --domain code      --target-tokens 3000000 --tokenizer-name smollm2
   python -m src.data.build_domain_heldout --domain books     --target-tokens 3000000 --tokenizer-name smollm2
   python -m src.data.build_domain_heldout --domain c4        --target-tokens 4000000 --tokenizer-name smollm2
   ```
   Domain slices go to `_runs/eval/heldout/<domain>_heldout_v1` by default.
4. **Rebuild the probe corpora.** The routing probe and expert masking read two
   fixed document sets drawn from `NeelNanda/pile-10k`, tokenized with SmolLM2:
   corpus A (`large500_smollm2`: 500 documents, 50 from each of 10 domains) and
   corpus B (`large500B_smollm2`: 485 further documents, disjoint from A). They are
   registered in `src/data/probe_corpus.py` and expected under
   `src/data/artifacts/corpus/`. The older probe scripts in `src/eval/probes/`
   default to a third, raw-text corpus of 40 documents (4 from each of 10
   domains); its command is the last one below. Rebuild them with:
   ```bash
   # corpus A, raw text
   python scripts/data/prepare_corpus.py \
       --out src/data/artifacts/corpus/pile10k_d50_b500_s512.json \
       --n-docs 500 --min-chars 4096 --max-per-domain 50 \
       --domains "ArXiv,Wikipedia (en),DM Mathematics,FreeLaw,Github,OpenWebText2,Pile-CC,PubMed Central,StackExchange,USPTO Backgrounds"
   # corpus B, raw text (same selection, skipping every document already in A)
   python scripts/data/build_corpus_b.py \
       --corpus-a src/data/artifacts/corpus/pile10k_d50_b500_s512.json \
       --out src/data/artifacts/corpus/pile10k_B_d50_b500_s512.json
   # SmolLM2-tokenized versions, read by the probe and masking commands
   python -m src.data.retokenize_probe_corpus \
       --corpus-path src/data/artifacts/corpus/pile10k_d50_b500_s512.json \
       --tokenizer-repo src/data/artifacts/tokenizer/smollm2 \
       --out src/data/artifacts/corpus/pile10k_d50_b500_s512.smollm2_v2.json
   python -m src.data.retokenize_probe_corpus \
       --corpus-path src/data/artifacts/corpus/pile10k_B_d50_b500_s512.json \
       --tokenizer-repo src/data/artifacts/tokenizer/smollm2 \
       --out src/data/artifacts/corpus/pile10k_B_d50_b500_s512.smollm2_v2.json
   # 40-document corpus, raw text (default of the older probe scripts)
   python scripts/data/prepare_corpus.py \
       --out src/data/artifacts/corpus/pile10k_d4_b40_s512.json \
       --n-docs 40 --min-chars 4096 --max-per-domain 4 \
       --domains "ArXiv,Books3,DM Mathematics,FreeLaw,Github,OpenWebText2,Pile-CC,PubMed Central,StackExchange,USPTO Backgrounds"
   ```
   Check the results against the files used in the paper. The raw-text files of
   corpus A and of the 40-document corpus must match byte for byte:

   | File | sha256 |
   |---|---|
   | `pile10k_d50_b500_s512.json` | `94ae1521aa5f018bd03879f22720e0b3d0d1ebdafc5f8a4d132ecd3cc0745a77` |
   | `pile10k_d4_b40_s512.json` | `863981870b55c76d62a733805117802de1048e7d0205d18a280b1b0597148696` |

   Corpus B's file records a text description of how it was built, which differs
   in wording from the paper's copy, so compare its documents instead:
   ```python
   import hashlib, json, sys
   print(hashlib.sha256(json.dumps(json.load(open(sys.argv[1]))["docs"]).encode()).hexdigest())
   ```

   | File | sha256 of documents |
   |---|---|
   | `pile10k_B_d50_b500_s512.json` (485 docs) | `07267455865ce386e4765f30cd49e5fcbd5c63395be345837881492dbc39b19a` |

   The tokenized files record the time they were made (`retokenized_at_utc`), so
   their file hashes always differ; what must match is the token ids of every
   document, in order. Hash them with:
   ```python
   import hashlib, json, sys
   docs = json.load(open(sys.argv[1]))["docs"]
   print(hashlib.sha256(json.dumps([d["token_ids"] for d in docs]).encode()).hexdigest())
   ```

   | File | sha256 of token ids |
   |---|---|
   | `pile10k_d50_b500_s512.smollm2_v2.json` (`large500_smollm2`, 500 docs, 2,830,118 tokens) | `5e8df84ddf68afd40e9124f35df48cec70f235b82a03b95b0421c7ac0a1e0726` |
   | `pile10k_B_d50_b500_s512.smollm2_v2.json` (`large500B_smollm2`, 485 docs, 2,299,892 tokens) | `fabb1ab9adcbea52cdbd33e98ea1b190b148a0ec4544055f683fbd37afd0ceb8` |

## Training

### Paper models

| Name in the paper | Run id | (experts E, loop-block layers D, loop passes L) | Attention across passes |
|---|---|---|---|
| Base-tied | S1 | (8, 8, 2) | shared |
| proto-Foil-3 | S2 | (16, 4, 4) | shared |
| proto-Foil-2 | S3 | (32, 2, 8) | shared |
| proto-Foil-1 | S4 | (64, 1, 16) | shared |
| Base | U1 | (8, 8, 2) | one set per pass |
| Foil-3 | U2 | (16, 4, 4) | one set per pass |
| Foil-2 | U3 | (32, 2, 8) | one set per pass |
| Foil-1 | U4 | (64, 1, 16) | one set per pass |

Run ids are defined in `src/pretrain/ablation_names.py`; shapes are in
`src/model/config_registry.py` (`ABLATIONS` for S/N/P/L runs, `ATTENTION_VARIANTS`
for U/UL runs, which reuse the S1–S4 and L-series shapes with one attention set
per loop pass). Every run's full plan (recipe, token budget, schedule,
checkpoint cadence) is built in `src/pretrain/recipe.py` and collected in
`RUN_LIST`.

### Launch commands

The launch command for every registered run is generated, not written by hand.
The output lists all registered runs; the first 9 are early trial runs that are
not in the paper.

```bash
python -m src.pretrain.print_run_commands --repo "$PWD" --markdown > run_commands.md
```

Each entry is headed by the run id (e.g. `U4 | ...`) followed by its `torchrun`
command (8 GPUs per node by default; `--nproc` changes it; `--data` points at a
different tokenized directory). Copy the entry for the run you want.
W&B logging reads the entity from the `WANDB_ENTITY` environment variable.

### 100B-token continuation

The 100B runs (`<run>_100B`, e.g. `U1_100B` … `U4_100B`) continue a finished
20B run from its step-45,000 checkpoint, taken before the learning-rate decay.
Their plans come from `plan_continuation` in `src/pretrain/recipe.py`, which
inherits everything from the source run except the name, step count, checkpoint
schedule and warmup. They appear in the same `print_run_commands` output; their
starting checkpoint is filled in from a path template:

```bash
python -m src.pretrain.print_run_commands --repo "$PWD" \
    --continuation-source '<root>/{run}/checkpoint/step-{step}'
```

`{run}` and `{step}` are filled in per run. The default template points at
`<repo>/_runs/{run}/checkpoint/step-{step}`.

## Evaluation

Downstream tasks (lm-evaluation-harness) on a Hugging Face-format checkpoint:

```bash
python -m src.eval.run_eval \
  --checkpoint-dir <checkpoint-dir> \
  --tokenizer-name smollm2 --max-length 4096 \
  --tasks squad_completion,coqa --num-fewshot 0 \
  --batch-size auto --log-samples \
  --include-path scripts/eval/tasks \
  --out <out-dir>/results.json
```

Held-out validation loss uses the same entry point with `--val-tokenized-dir`,
`--val-seq-len` and `--no-lm-eval`. `--include-path scripts/eval/tasks` makes the
custom `wsc273_nsmirror` task available (adapted from lm-evaluation-harness; see
`NOTICE`).

## Routing probe

One forward pass per checkpoint; writes per-layer routing metrics (`.jsonl`,
`.npz`) and every router's raw scores over all experts (`.router_logits.npz`):

```bash
python -m src.eval.probes.offline_cross_arch_probe \
  --checkpoint-dir <checkpoint-dir> \
  --out       <out>/<name>.jsonl \
  --dump-npz  <out>/<name>.npz \
  --dump-router-logits <out>/<name>.router_logits.npz \
  --logits-tokens 0 --logits-seed 0 --logits-dtype float16 \
  --corpus large500_smollm2 --device cuda --probe-dtype float32 \
  --n-documents 500 --doc-chunk 50 \
  --c-null-tokens 100000 --c-null-seed 0 \
  --hidden-subsample-tokens 0 --hidden-seed 0
```

## Expert masking

```bash
python -m src.eval.expert_masking.run \
  --checkpoint-dir <checkpoint-dir> \
  --run-label <name> --out-root <masking-root> \
  --corpus-a large500_smollm2 --corpus-b large500B_smollm2 \
  --tiers 5,10,20 --n-random 20 \
  --n-documents-a 500 --n-documents-b 485 --doc-chunk 50 \
  --probe-dtype float32 --device cuda

# per-token reassignment counts, run after the command above
python -m src.eval.expert_masking.backfill_reassign \
  --product-dir <masking-root>/<name> \
  --checkpoint-dir <checkpoint-dir> \
  --doc-chunk 50 --probe-dtype float32 --sets all --device cuda
```

## License

Apache License 2.0 (`LICENSE`). Third-party attributions are in `NOTICE`.
