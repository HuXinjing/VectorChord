# GitHub branch review, 2026-11-16

Repository: `HuXinjing/VectorChord`. Snapshot base: release `5801ab1`. No remote branch has been deleted or renamed.

The running `hopper-8189570-20261101` image maps to commit `8189570` on the release ancestry. Release tip `5801ab1` additionally merges the single-vector companion; it changes no existing Rust scorer files relative to that image commit. New scorer work belongs on this release line, not the divergent fork default branch.

| Branch | SHA | Commits unique to branch | Recommendation |
| --- | --- | ---: | --- |
| `feat/single-vector-tenant-gpu-20261102` | `7172998` | 0 | Deletion candidate after approval: PR #1 merged |
| `feature/adaptive-continuous-tilemaxsim` | `9fbb210` | 0 | Deletion candidate after approval: fully included in release |
| `feature/portable-accelerator-backends` | `835a1a2` | 0 | Deletion candidate after approval: fully included in release |
| `feature/quantized-tilemaxsim-research` | `03ab2fd` | 343 | Preserve/archive and review unique changes before deletion |
| `feature/tutti-io-pipeline` | `794e763` | 325 | Preserve/archive and review unique changes before deletion |
| `fix/raw-fp8-shard-storage-20261026` | `e4009a7` | 0 | Deletion candidate after approval: fully included in release |
| `fix/tilemaxsim-microbatch-result-slots-20261029` | `940a516` | 0 | Deletion candidate after approval: fully included in release |
| `integration/upstream-main-tilemaxsim-v2` | `4d2ce60` | 0 | Deletion candidate after approval: fully included in release |
| `main` | `6f2317b` | 324 | Keep: default branch; divergent history |
| `perf/hopper-fused-maxsim-release-20261101` | `5801ab1` | 0 | Keep: active release line |
| `perf/tilemaxsim-cpu-request-path-20261031` | `8189570` | 0 | Tag/archive the current deployed image commit before considering deletion |
| `upstream/maxsim-exact-rerank` | `129d1a3` | 1 | Keep for review: one patch unique even after git cherry comparison |
| `upstream/maxsim-planner-cost` | `ca157ea` | 1 | Keep for review: one patch unique even after git cherry comparison |

The quantized-research and tutti branches have hundreds of commits absent from this release. An ancestry check alone cannot classify their work as obsolete. The two upstream branches each retain one non-equivalent patch. Do not mass-delete these or change the default branch as part of a cache-recovery rollout.

Suggested order: preserve deployment tags; approve the fully merged ancestor list; delete only approved branches; separately review/archive divergent research and upstream patches. The temporary local implementation worktree is independent of remote branch deletion.
