# SmolLM2 tokenizer files -- provenance

**Purpose**: bundle SmolLM2's tokenizer files directly in this repo -- tokenization (data prep,
on any machine including a training cluster) should not depend on Hugging Face Hub network
access or an HF token.
These are tokenizer files (vocabulary + BPE merges + special-token config), NOT model weights.

**Context**: the paper's models use SmolLM2 tokenization; earlier runs used the Llama-3
tokenizer, which is not bundled (see `src/data/tokenizer.py`).

## Source

- **Repo**: `HuggingFaceTB/SmolLM2-1.7B` (Hugging Face Hub)
- **Downloaded**: 2026-09-12, via `huggingface_hub.hf_hub_download`.
- **Revision**: commit `effd688a12921b4cc83e3312b6feb579f70f9c71` (pinned via
  `HfApi.model_info("HuggingFaceTB/SmolLM2-1.7B").sha`, not `main` -- so this provenance record
  stays valid even if the repo's default branch moves later).

## Files and sha256 (full-file hashes, not truncated)

| File | sha256 | Size |
|---|---|---|
| `tokenizer.json` | `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c` | 2,104,556 bytes |
| `tokenizer_config.json` | `4bb9af56a342753d39374f4016a16574cab299fe088e896f425ce3c433f61424` | 3,658 bytes |
| `special_tokens_map.json` | `e786b595b9a23148bf1630df78d9037a048ea671e48bfd3549a1e3c233742bb3` | 831 bytes |

## Verified fields (actually loaded and checked, not assumed from the model card)

- `len(tokenizer)` (total vocab, what a model's embedding/output layer must match) = **49152**,
  matches the `SMOLLM2_VOCAB_SIZE` constant in `src/model/config_registry.py`.
- `tokenizer.vocab_size` (base vocab attribute) is also 49152 here -- unlike Llama-3's fast
  tokenizer, SmolLM2's `vocab_size` attribute and `len(tokenizer)` agree (no added-special-tokens
  discrepancy to catch in this case; still worth stating explicitly rather than assuming the two
  are always the same, since Llama-3's case is exactly the counterexample that made this project
  distrust `.vocab_size` alone).
- `bos_token_id == eos_token_id == 0`, both mapping to the single special token
  `<|endoftext|>` -- SmolLM2 (GPT-2-style tokenizer) uses ONE token for both roles, unlike
  Llama-3's separate `<|begin_of_text|>` (128000) / `<|end_of_text|>` (128001). This is a real
  tokenizer-family difference, not a bug in these files -- code that frames documents as
  "BOS + text + EOS" produces the same token id at both ends for SmolLM2, which is the correct,
  expected behavior for this tokenizer (see `src/data/tokenizer.py`'s per-name framing).

## What loads these files

`src/data/tokenizer.py` selects a tokenizer BY NAME (`"llama3"` or `"smollm2"`, required, no
default -- see that module's docstring for why a silent default would be exactly the kind of
mistake this project's "semantic parameters have no default" discipline exists to prevent).
Selecting `"smollm2"` defaults to this local directory; passing an explicit HF repo id still
works if ever needed (e.g. a newer SmolLM2 revision).
