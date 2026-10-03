# ---------------------------------------------------------------------------
# VENDORED SNIPPET — DO NOT EDIT BELOW THIS HEADER.
#
# Source: an earlier, separate diagnostics code base (read-only reference);
# original module path collect/schema.py (function `sha8` only -- NOT the whole file: schema.py's TraceMeta/
#   TraceGroup dataclass machinery is collection-layer plumbing this project's
#   training-time metrics module has no use for; vendoring the whole file
#   would add entities without need. Only this one function is needed
#   (input_ids_sha8 provenance hashing for probe-corpus traceability), and it
#   is copied character-for-character, unmodified.)
#
# In the development repo a sync test (not shipped here) compares this
# function's code with the source function and fails on drift.
# ---------------------------------------------------------------------------

from __future__ import annotations

import hashlib

from torch import Tensor


def sha8(t: Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:8]
