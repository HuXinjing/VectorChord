# Read/upload decoupling experiment (2026-11-17)

The upload budget is independently configurable with `--io-upload-batch-mib` (1 through 4096 MiB). Omission preserves the read batch budget. `--io-pipeline` remains disabled by default. Neither tensor shapes nor native kernels changed.

The bounded producer reads and validates immutable objects using the existing store. The consumer coalesces verified objects, then uses the existing upload/scoring and GPU admission paths. Upload and scoring remain serialized. Payload staging is bounded by the upload budget plus three read budgets, with allowance for indivisible oversized tensors; host cache and GPU capacity limits remain separate.

## Controlled GPU 2 measurements

Three trials per cell, 16,384 canonical FP16 tensors, dimension 320, 7,832,089,600 payload bytes, 28 GiB GPU cache, 2 GiB workspace, 16 GiB software host cache, 16 readers, automatic 4096-candidate quantum. This measures the engine path rather than Agent or HTTP latency. Read batches were 64 MiB. Every process began with an empty GPU cache. There were no GPU evictions.

Private copies on the same NVMe filesystem isolated cold-page eviction from production objects. Cold trials verified less than 1% page residency and at least 90% physical reads; page-cache trials verified over 99% residency and under 1% physical reads. Software-cache-hit trials performed no tensor file reads. No global page cache was dropped.

Median elapsed seconds:

| Initial source | Original path | Overlap U=64 MiB | U=256 MiB | U=512 MiB | U=1024 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Software host cache | 2.147 | 3.233 | 2.664 | 2.288 | 2.335 |
| OS page cache | 5.332 | 4.673 | 4.867 | 5.384 | 5.468 |
| Cold NVMe | 5.839 | 5.076 | 5.416 | 5.599 | 6.285 |

Large upload groups recover most of the fragmented-upload penalty on software-cache hits, but delay consumption of cold reads. They do not establish a universal improvement. Keep 64 MiB for cold overlap; do not change production defaults based on this experiment. Host-transfer timing variability was visible (its cause was not isolated), including an original-path host-hit range of 1.955 to 4.598 seconds. Backend API timings include CPU copies and synchronization and are not pure CUDA kernel or PCIe timings.

All 45 samples produced the identical score SHA-256 `7dd8f805059a70f0db03e1c0ba5fac7bf35c28b92a4b08b000b8fffddf3e9c57`. Every sample uploaded exactly the payload size. The private corpus was removed and the production container identity was unchanged. GPU 7 remained untouched.

## Optional host-hit routing

`--io-pipeline adaptive` checks whether all GPU-missing descriptors are present in the software host cache. If so, it retains the original whole-quantum path. Otherwise it uses overlap with the configured budgets. This is only a routing hint: resolution, validation, policy updates, admission and cleanup still execute normally. The hint does not increment cache hit counters or change ownership/frequency. Partial hits and concurrent eviction remain safe.

CPU validation: 135 library tests and 6 control tests passed, including partial hits, score equality, capacity pressure, corrupted objects, release after errors and CLI bounds. Clippy passed for all CPU targets. CUDA example compilation passed. Six additional GPU 2 trials verified the adaptive host-hit bypass: original median 1.561 seconds, adaptive median 1.603 seconds (about 2.7% higher; three trials each). Both performed four upload calls, zero tensor reads and produced identical score hashes. The earlier fragmented path performed 120 upload calls. This small follow-up validates bypass behavior, not a statistically established speedup. Cold adaptive routing is covered by CPU tests but has not been separately GPU benchmarked. This mode is optional and has not been deployed.


## Provenance and release recommendation

The 45-cell-sample run used implementation commit `95c88c4`, based on Hopper release `3adce280416602f20e40f479ec87d27ff7680353`. Its diagnostic binary SHA-256 was `34f7f2aed7ee6cb38f08f4e24fd97df475788eb684e3bcea2045a67c66c47ff0`. The six host-hit follow-up samples used the source now committed as `981ef81`; diagnostic binary SHA-256 was `3821c613439cab1f4194baf7e61d01a09dc5d4efd4977eb5ae030f45e3b276b0`. Both used frozen manifest SHA-256 `55bf61ed9e932050301b274113c61bdeb9a47af5666509d1c8b2867d436bf68d`, GPU 2 on an H200 host, and the existing CUDA 12.6 runtime. GPU eviction, production queueing, concurrent tenants and end-to-end retrieval were outside this measurement.

The diagnostic wrapper timed existing upload and scoring APIs. Private corpus manifests and raw logs remain in the local restricted experiment archive; they are intentionally not published. Consequently this report is evidence for the stated controlled workload, not a portable public-corpus benchmark or a claim that all production workloads improve.

Retain independent upload budgeting and adaptive routing as opt-in experimental controls. Do not raise the default upload budget, enable overlap globally, or replace the production image based on these samples. Before enabling adaptive mode in production, separately measure mixed host-hit/miss workloads, GPU capacity pressure, concurrency, cancellation and recovery on the intended deployment. The existing cache-eviction metadata scan under capacity pressure is a separate unresolved issue; this patch does not resolve it.
