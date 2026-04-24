# Pinned-Host-Pictures Matrix — 2026-04-24

Two-GPU cross-check of the pinned-host-fetch-ring change
(commit `9431ea6d`) on the same developer box, same clips, same
binary. Follow-up to `pinned-host-pictures-2026-04-24.md`
(RTX 3080 single-GPU run earlier in the day).

## Hardware

| Dev | GPU       | Arch      | sm_ | VRAM |
|----:|-----------|-----------|-----|-----:|
|  0  | RTX 5070  | Blackwell | 120 | 12 GB |
|  1  | RTX 3080  | Ampere    |  86 | 10 GB |

Driver 595.58.03, CUDA 13.2. PCIe slot topology and motherboard
unchanged from the morning run.

## Results (bare wall-clock, best-of-3, 300 frames, `crowd_run`)

| GPU      |    480p |   1080p |   2160p |
|----------|--------:|--------:|--------:|
| RTX 5070 |  671.14 |  388.10 |  121.46 |
| RTX 3080 |  738.92 |  402.68 |  122.15 |

| GPU      | 480p VMAF | 1080p VMAF | 2160p VMAF |
|----------|----------:|-----------:|-----------:|
| RTX 5070 |   18.7483 |    18.7538 |    23.5754 |
| RTX 3080 |   18.7483 |    18.7538 |    23.5754 |

Bit-exact across both GPUs at every tier.

## Observations

**RTX 3080 leads the RTX 5070 at every tier** by +10% / +4% / +1%.
Reversal from the pre-pinning lever-2 matrix (`lever2-2026-04-19.md`),
where the 5070 beat the 4070 at 2160p by ~12% thanks to Blackwell's
advantage in the still-GPU-bound regime.

Post-pinning, `cuMemcpy2DAsync` is no longer the bottleneck at any
tier (2160p API-time share 68.8% → 3.4%; see
`pinned-host-pictures-2026-04-24.md`). The dominant residual is
`cuLaunchKernel` overhead (~23% of API time at 2160p post-pinning,
equal absolute cost to pre-pinning). Launch cost does not scale with
GPU compute architecture — it scales with the host-side driver path
— so the 5070's extra Blackwell compute buys nothing in this regime.
The 3080 edging ahead is within run-to-run variance at 2160p (~1%)
but persistent at 480p/1080p (~4-10%), likely driver overhead on the
newer card or system-variability noise we haven't isolated.

The practical reading: we've lifted the ceiling for the PCIe-bound
case but we've also exposed that Scope-A style CPU-side work
(context-bind-once + memory pool) and CUDA graphs are now the
highest-ROI next targets — they'd claw back launch overhead that is
now *the* bottleneck at every tier on modern hardware.

## Raw fps per run

```
rtx5070  480p:  641.03  671.14  668.15
rtx5070 1080p:  384.62  369.00  388.10
rtx5070 2160p:  121.36  120.34  121.46
rtx3080  480p:  738.92  722.89  738.92
rtx3080 1080p:  400.53  402.68  395.78
rtx3080 2160p:  122.15  121.07  121.51
```

Artifacts in `benchmark/results/2026-04-24-pinned-matrix/`.
