"""Vendored copies of metric-definition functions from an earlier diagnostics code base.

Every file in this package (except this one) is a verbatim copy of a file
(or, for `sha8.py`, a single function) from an earlier, separate diagnostics
code base (read-only reference, not a submodule). See each file's header
comment for its original module path.

Why vendored: this is the repo that ships to the training cluster; the sibling
repo does not travel there. Training-time metrics must be computable from this
repo alone, so the metric *definitions* this project already verified are
copied here rather than reimplemented (reuse the existing,
tested definition; do not invent a second one that could silently drift
from the original's semantics).

Do not edit these files by hand. In the development repo a sync test (not
shipped here) compares them with the source repo and fails on drift.

If a needed function from that code base is not yet vendored here, add it the
same way (copy verbatim, prepend the same header format, add it to the
sync-check test) rather than reimplementing it inline in this project's
own metrics code.
"""
