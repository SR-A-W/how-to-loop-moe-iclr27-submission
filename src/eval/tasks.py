"""The LT2 (arXiv 2605.20670) zero-shot evaluation task list -- source of truth for `run_eval.py`.

Provenance: the LT2 training recipe's evaluation section, itself sourced from
the LT2 paper's Appendix C "Evaluation" section, verified against paper text
(Table 2 shows fewer columns than the full set -- that is the paper's own
selective display, not a smaller task list; the recipe is explicit that all 11
are the target set).

Each paper task name is mapped to its LM-Evaluation-Harness registered task
name (verified by inspecting `lm_eval==0.4.12`'s bundled task YAML
`task:` fields directly -- not assumed from the paper's prose, which uses
casual names like "SocialIQA" that do not match the harness's registered
`social_iqa`, or "COPA" which is nested under `super_glue/copa/` but
registers as the bare `copa`). A wrong mapping here would silently evaluate
a different task from what the paper reports, producing a number that is
real but not comparable to the LT2 baseline -- exactly the kind of
plausible-looking-but-wrong result this project's discipline exists to
prevent.

    paper name (recipe doc)  ->  lm_eval task name  ->  yaml file inspected
    HellaSwag                    hellaswag              lm_eval/tasks/hellaswag/hellaswag.yaml
    BoolQ                        boolq                  lm_eval/tasks/super_glue/boolq/default.yaml
    PIQA                         piqa                   lm_eval/tasks/piqa/piqa.yaml
    SocialIQA                    social_iqa             lm_eval/tasks/siqa/siqa.yaml
    WinoGrande                   winogrande             lm_eval/tasks/winogrande/default.yaml
    OpenBookQA                   openbookqa             lm_eval/tasks/openbookqa/openbookqa.yaml
    ARC-Easy                     arc_easy               lm_eval/tasks/arc/arc_easy.yaml
    ARC-Challenge                arc_challenge          lm_eval/tasks/arc/arc_challenge.yaml
    RACE                         race                   lm_eval/tasks/race/race.yaml
    CommonsenseQA                commonsense_qa         lm_eval/tasks/commonsense_qa/default.yaml
    COPA                         copa                   lm_eval/tasks/super_glue/copa/default.yaml
"""

from __future__ import annotations

# Registered lm-eval-harness task names, in the order the recipe doc lists
# them. All eleven, run zero-shot (num_fewshot=0) as the recipe's "zero-shot
# downstream tasks" -- the paper reports Table 2 as zero-shot, and the recipe doc's eval
# section makes no mention of any few-shot variant for these.
LT2_ZERO_SHOT_TASKS: tuple[str, ...] = (
    "hellaswag",
    "boolq",
    "piqa",
    "social_iqa",
    "winogrande",
    "openbookqa",
    "arc_easy",
    "arc_challenge",
    "race",
    "commonsense_qa",
    "copa",
)

NUM_FEWSHOT = 0
"""Zero-shot, as the recipe's downstream tasks require -- not a default. Passed
explicitly at every `simple_evaluate` call site rather than relying on each
task yaml's own default, since a few task configs upstream carry a non-zero
default and lm-eval applies the CLI/API argument as an override -- being
explicit here means this script's behavior does not silently drift if any
task's own default changes in a future lm-eval release.
"""
