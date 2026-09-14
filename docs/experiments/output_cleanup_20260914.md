# Output cleanup — 2026-09-14

The user requested removal of obsolete experiment outputs. Removed 94 old trial
and temporary-check directories plus 63 obsolete root-level log/preview files
from `PORT-GS/runs/`, about 9.70 GiB of allocated file storage. The output tree
now occupies approximately 2.4 GiB. No active PORT-GS Python training process
was found before deletion. The retained comparison manifests had no references
into the deletion targets.

The exact 157 paths and their pre-deletion allocated sizes are recorded in
[output_cleanup_20260914.json](output_cleanup_20260914.json). All listed paths
were deleted permanently; ignored experiment outputs are not stored in Git.

Retained directories:

- `hashgrid_validation_20260914`: current six-scene results and checkpoints.
- `rank512_validation_20260914`: the six-scene learned-anchor512 comparison.
- `full_benchmark_20260913`: historical benchmark results, comparison inputs,
  and queue evidence, including its pre-existing failed status.
- `research_20260912`: baseline32 results, source archives, and the Cat r1
  checkpoint required by current transport/receiver regression tests.
- `hashgrid_preflight`: operator and receiver audit reports; the two-step
  training checkpoint and one-frame reload evaluation were deleted.

Compact root-level JSON/CSV summaries, patches and research scripts are retained.
Source code, Git history, documents, uploaded papers, datasets, dependencies and
other method projects were not cleanup targets. Older documentation may link to
removed trial outputs; those links are historical evidence references, not
instructions to recreate deleted directories.
