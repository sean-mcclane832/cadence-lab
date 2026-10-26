# CPU vs CUDA on a consumer GPU (v1)

Same composed brain, same life, three backends. `Brain.compose(4, 2, modules=(N,), arousal=True)`
is composed and saved on the CPU, then loaded with `Brain.load(path, backend=..., precision=...)`
so every arm starts from identical parameters. Task: 4 one-hot cues, 2 actions, the paying action
is `cue % 2` (+1 / -1), one `live` call per moment. One untimed warm-up moment per arm (JIT, CUDA
init) on a separate load. Cadence 0.89.0, torch 2.11.0+cu128, i7-9700K, RTX 2060 SUPER 8 GB.
Receipts: `runs/backend_timing/receipt.json`, `receipt_1536.json`.

| modules | moments | seeds | CPU (NumPy+Numba f64) | CUDA float64 | CUDA float32 |
| --- | --- | --- | --- | --- | --- |
| (64,) | 300 | 0, 1 | 0.87-1.16 ms | 27.1-28.7 ms | 27.5-27.6 ms |
| (512,) | 300 | 0, 1 | 4.3 ms | 27.8-28.3 ms | 28.5-29.0 ms |
| (1536,) | 100 | 0, 1 | 90-91 ms | 38.5-40.3 ms | 67.3-67.5 ms |

(ms per moment, range over seeds)

- Every backend made the same choices: identical sweep counts and hit rates per seed.
- Small brains are launch-bound on the GPU: ~28 ms per moment whatever the size, 6-30x slower
  than the CPU. This matches the Cadence docs (CPU with Numba wins on small continuing brains).
- The GPU only wins at the largest size tried (1536): CUDA float64 is 2.3x faster than the CPU.
- **Surprise: float32 was slower than float64 on this card** (67 vs 39 ms at 1536), even though
  the 2060 SUPER's float64 rate is a small fraction of its float32 rate. Unexplained; likely host
  conversions or a path difference in the float32 code, not arithmetic. Worth profiling before
  claiming anything.

Limits: two seeds, short lives, one task, wall-clock on a desktop with other apps open, no
profiler. Not a throughput benchmark; the upstream `benchmarks/system1_cuda` protocol is the
real one.
