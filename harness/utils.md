# Harness utils — extend these, do not paste copies

Host/device plumbing lives here. Later parts should add to these files instead of growing a second memcpy/timer/JSON path.

## Device memory (`device_memory.h`)

RAII buffer + copies. `cudaFree(nullptr)` is a no-op, so empty `DeviceUnique` is safe.

| Helper | Use |
|---|---|
| `DeviceAlloc<T>(n)` | `cudaMalloc`, unique_ptr deleter |
| `DeviceFill(ptr, byte, n)` | `cudaMemset` |
| `CopyHostToDevice` / `CopyDeviceToHost` | raw or `DeviceUnique` + `span` |
| `DeviceFromHost(span)` | alloc + H2D |
| `HostFromDevice(ptr, n)` | D2H into `vector` |

Need a new transfer shape (async stream, 2D, pitched)? Add it here. Do not open-code `cudaMemcpy` in a part.

## Already in the harness (do not rebuild)

- Timing: `TimeCudaLaunch` / `TimeHost` (`timing.h`) — CUDA events, warmup 3, reps 20
- Results JSON: `WriteJson` (`results.h`) — same schema for every part
- Env: `CaptureEnv` — clocks, git, GPU name
- Verify gate: `RecordVerifyFullOk` / `ClearVerifyFull` / `VerifyFullPassed`
- Stats: `harness/stats/` KS, χ², moments
- RNG: `harness/rng/philox.h` (CPU). Device XORWOW is cuRAND, not a home-rolled generator
- Sample dump: `dump.h` (JSON metadata + float4 binary)

## Libraries (FetchContent, already pinned)

nlohmann/json, CLI11, {fmt}. Device algorithm: `cuda::std::` (libcu++ with the toolkit). cuRAND for device RNG. Hand-roll only the science (special functions, kernels, stats).
