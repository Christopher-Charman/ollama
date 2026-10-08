# PowerPC Xeon CPU inference variants — controlled experiment

**Status: DRAFT / SOURCE-ONLY / NOT DEPLOYED.**

This opt-in branch adds two CPU build presets alongside Ollama's existing `cpu` preset. It does **not** change production defaults, replace the Ollama binary, repoint OpenWebUI, modify the model store, or alter the running llama.cpp router.

## Why test inside Ollama?

Ollama's API/model-manager/scheduler already launches a version-pinned llama.cpp server backend. Using a host-tuned backend inside Ollama may retain native Ollama API semantics, existing GGUF management and model lifecycle without incurring an external compatibility shim. That is a **hypothesis** until an instrumented live release passes end-to-end parity tests.

The current fork's `LLAMA_CPP_VERSION` is `b11351` (read from `main` on 2026-10-09); the separately forked `Christopher-Charman/llama.cpp` is not automatically the same source revision. The configured FetchContent URL still points at `ggml-org/llama.cpp`. The Ollama compatibility code under `llama/compat/` applies patch hooks to selected published models; this is not optional unless model compatibility is independently demonstrated. **Do not rebase or substitute the separate fork into the Ollama build without a clean compatibility-patch validation and regression suite.** `OLLAMA_LLAMA_CPP_SOURCE` skips automatic patch application; using it unprepared is not accepted.

## Presets

| Preset | CPU backend strategy | Inherits | Output build directory |
|---|---|---|---|
| `cpu` (unchanged control) | all supported CPU variants; `GGML_NATIVE=OFF`, `GGML_OPENMP=OFF` | default | `build/llama-server-cpu` |
| `cpu_native_xeon` | exact host CPU native backend; `GGML_CPU_ALL_VARIANTS=OFF`, `GGML_NATIVE=ON`, `GGML_OPENMP=OFF` | cpu | `build/llama-server-cpu_native_xeon` |
| `cpu_native_xeon_openmp` | same native backend with `GGML_OPENMP=ON` | cpu_native_xeon | `build/llama-server-cpu_native_xeon_openmp` |

Both new presets are explicit experiments, not a portable release. No runtime configuration or released build may select either automatically. A compiled host-native binary may be incompatible with a different CPU.

## Build admission

- Verify the actual managed PowerPC Unix identity `csh3280350` / UID `2257347` and execution namespace before acting. Do not confuse the provider SSH gateway with the managed Passenger runtime, or use Evenio.
- Check the executable CMake/GCC/Go toolchain, the host's CPU flags and permitted core affinity, and Linux/glibc compatibility. The current Ollama fork declares `go 1.26.0`; availability of an adequate Go compiler **has not** been established by the fork.
- Check the live Concurrency Ledger and sibling worker intentions immediately before a compile or substantial RAM/CPU test. Do not interfere with concurrently running PowerPC workloads.
- Work in isolated source/build directories; keep existing `/home/storage/781/4477781/user/webapp/miniconda/bin/ollama` and model files intact.
- Start with the fork's pinned llama.cpp source and its original compatibility patches (do not mix backend version changes with compiler flags). Verify the fetched and patched checkout, and run compile and integration tests independently.
- Validate configure: `cmake -S llama/server --preset cpu_native_xeon`. Validate build: `cmake --build build/llama-server-cpu_native_xeon --target llama-server --parallel 3`. Repeat for `cpu_native_xeon_openmp` only once the first candidate passes. Do not assume these build commands have succeeded merely because the presets are syntactically valid.
- Inspect `ldd` on generated binaries, CPU ISA, dynamic backend loading and model startup before attempting any service launch. No production service may point to the candidate without explicit run acceptance.

## Benchmark acceptance

Owner: [repository-prime issue #103](https://github.com/Christopher-Charman/repository-prime/issues/103).

Benchmark **three build variants of Ollama's pinned backend** on the same host, same GGUF bytes, model/template, generation settings, context, load state, core affinity and resource profile. Then compare the best parity-clean variant against standalone llama.cpp as a separate engine-level comparison.

1. Establish an idle baseline and preserve Ollama `:11434`, Qwen `:18084`, BGE-M3 `:18085`, supervised router `:18086`, and OpenWebUI `:18080`. No provider switch or kill/stop of these services is included in this PR.
2. Evaluate thread counts 1, 2, 3, 4, 5 and 6 (host currently exposes six logical CPUs / three physical cores), plus context 2K/4K/8K when memory permits. Record affinity and neighbour contention; do not assume more threads are faster.
3. Match prompt bytes, token limits, sampler, raw/template path, cache policy, warm state, quantisation, model SHA and CPU-only processing. Separate engine latency from HTTP/API overhead.
4. Record cold-model load latency; warm time-to-first-token (stream), prompt tokens/sec, decode tokens/sec, full-request wall time, P50/P95/P99 when sample counts support them, process RSS/PSS, system `MemAvailable`, swap-in/out and unexpected errors. Use >=7 warm repetitions in interleaved or randomized order and preserve raw JSON/TSV receipts (private if containing model text or private traces).
5. Representative workloads: Gemma 270M Q8_0, Qwen Coder 1.5B Q4_K_M, Local Agent Gemma 1B Q4_K_M, and BGE-M3 F16 embeddings; larger installed models require an independent memory admission decision.
6. Functional matrix: Ollama native API model inventory, metadata, pulls/creates/copies/removes, `/api/generate`, `/api/chat`, `/api/embed`, `/api/ps`, `keep_alive`, streaming, structured JSON/schema, tool calls, templates and stop tokens, 1K/4K/8K context, concurrency, failures/timeouts and inference repeatability.
7. OpenWebUI: completion, Local Agent tasks, RAG embedding, TTS, model discovery, title/task generation, restart and rollback. The previously recorded OpenWebUI index-5 provider-switch platform safety block remains binding; **do not reroute that denied operation**.
8. Compare candidate against unchanged Ollama baseline and standalone llama.cpp, with practical effect sizes and variability. Adopt only an improvement that is repeatable, material for actual workloads and parity-clean. Otherwise retain the existing build. No one engine is assumed to be faster for every model.

## Verified historical measurement, not an optimisation claim

On 2026-10-09 the live same-GGUF test (one cold + three warm) measured median warm decode for Gemma 270M at 37.95 tok/s (stock Ollama) vs 28.83 (standalone native router), while Qwen 1.5B measured 15.61 vs 17.28 respectively. This non-interleaved 3-run result is a hypothesis generator, not a robust winner or an evaluation of **these new unbuilt presets**.

## Deployment and rollback

Keep the live Ollama binary, its untouched model store, existing provider endpoints, original source pin and working service supervisor unchanged. Stage any successful candidate at a different path, port and supervision scope. Require measured performance, complete client-visible parity, operator-approved release and tested rollback before any production substitution. Security/platform permission denials are independent of user approval and must not be circumvented.
