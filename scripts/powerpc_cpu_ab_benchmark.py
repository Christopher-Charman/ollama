#!/usr/bin/env python3
"""Controlled warm inference benchmark, Python >=3.6, standard library only.

Safe by default: prints a plan. --execute requires a pinned blob SHA, owner UID,
an exclusive private output path, and a minimum memory headroom. Never changes
OpenWebUI provider, binaries, model files, or Ollama API configuration.
"""
import argparse
import hashlib
import io
import json
import os
import platform
import random
import re
import sys
import time
import urllib.request

OLLAMA = "http://127.0.0.1:11434/api/generate"
NATIVE = "http://127.0.0.1:18086/completion"
ALLOWED = ("gemma3:270m-core", "qwen2.5-coder:1.5b-instruct-q4_K_M")
PROMPT = "Explain in two sentences how a checksum detects changes to a file."
REQ_TOKENS = 48


def memory_kib():
    values = {}
    with open("/proc/meminfo", "r") as stream:
        for line in stream:
            parts = line.split()
            if parts[0].rstrip(":") in ("MemAvailable", "SwapFree"):
                values[parts[0].rstrip(":")] = int(parts[1])
    return values


def require_headroom(min_mib):
    values = memory_kib()
    if values.get("MemAvailable", 0) < min_mib * 1024:
        raise RuntimeError("MEMORY_HEADROOM_BELOW_GATE: available MiB=%d; required=%d" %
                           (values.get("MemAvailable", 0) // 1024, min_mib))
    return values


def payload(engine, model):
    if engine == "ollama":
        return {
            "model": model, "prompt": PROMPT, "raw": True, "stream": True,
            "keep_alive": "30s",
            "options": {"num_ctx": 4096, "num_predict": REQ_TOKENS,
                        "num_thread": 4, "num_gpu": 0, "temperature": 0,
                        "top_k": 1, "top_p": 1, "repeat_penalty": 1, "seed": 42}
        }
    return {"model": model, "prompt": PROMPT, "stream": True,
            "n_predict": REQ_TOKENS, "cache_prompt": False,
            "temperature": 0, "top_k": 1, "top_p": 1,
            "repeat_penalty": 1, "seed": 42}


def decode_stream(engine, lines, started, clock=time.monotonic):
    """Decode Ollama NDJSON or llama.cpp server-sent events without logging text."""
    chunks, final = [], {}
    ttft = None
    for raw in lines:
        line = raw.decode("utf-8").strip() if isinstance(raw, bytes) else raw.strip()
        if engine == "native":
            if not line.startswith("data: "):
                continue
            line = line[6:]
            if line == "[DONE]":
                break
        if not line:
            continue
        message = json.loads(line)
        if message.get("error"):
            raise RuntimeError("INFERENCE_ERROR: " + str(message["error"])[:120])
        piece = message.get("response", "") if engine == "ollama" else message.get("content", "")
        if piece:
            if ttft is None:
                ttft = round((clock() - started) * 1000, 3)
            chunks.append(piece)
        if message.get("done") or message.get("stop"):
            final = message
    if ttft is None or not final:
        raise RuntimeError("INCOMPLETE_STREAM_OR_ZERO_OUTPUT")
    return ttft, "".join(chunks), final


def measure(engine, model):
    obj = payload(engine, model)
    url = OLLAMA if engine == "ollama" else NATIVE
    req = urllib.request.Request(url, json.dumps(obj).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=110) as stream:
        if stream.status != 200:
            raise RuntimeError("NON_200_HTTP")
        ttft, output, final = decode_stream(engine, stream, started)
    elapsed = round((time.monotonic() - started) * 1000, 3)
    timings = final if engine == "ollama" else final.get("timings", {})
    gen = final.get("eval_count", 0) if engine == "ollama" else timings.get("predicted_n", 0)
    decode_ms = (final.get("eval_duration", 0) / 1000000.0 if engine == "ollama"
                 else timings.get("predicted_ms", 0))
    return {
        "engine": engine, "model": model, "wall_ms": elapsed,
        "ttft_ms": ttft, "generated_tokens": gen,
        "generation_tps": round(gen * 1000.0 / decode_ms, 3) if decode_ms else None,
        "prompt_tokens": final.get("prompt_eval_count") if engine == "ollama" else timings.get("prompt_n"),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "meminfo_after_kib": memory_kib(),
        "complete": True
    }


def plan_order(blocks, seed):
    rng = random.Random(seed)
    order = []
    for i in range(blocks):
        pair = ["ollama", "native"]
        rng.shuffle(pair)
        order.extend(pair)
    return order


def selftest():
    assert len(plan_order(4, 42)) == 8
    assert plan_order(4, 42) == plan_order(4, 42)
    assert payload("ollama", ALLOWED[0])["options"]["num_ctx"] == 4096
    assert payload("native", ALLOWED[0])["cache_prompt"] is False
    fake_o = [b'{"response":"OK","done":false}\n',
              b'{"response":"","done":true,"eval_count":1,"eval_duration":1000000000}\n']
    fake_n = [b'data: {"content":"OK","stop":false}\n',
              b'data: {"content":"","stop":true,"timings":{"predicted_n":1,"predicted_ms":1000}}\n']
    for engine, lines in (("ollama", fake_o), ("native", fake_n)):
        ttft, txt, final = decode_stream(engine, lines, 0, clock=lambda: 0.01)
        assert ttft == 10 and txt == "OK" and final
    try:
        decode_stream("ollama", [b'{"response":"OK","done":false}\n'], 0, clock=lambda: 0.01)
    except RuntimeError as error:
        assert "INCOMPLETE" in str(error)
    else:
        raise AssertionError("incomplete stream must fail")
    print("SELFTEST_PASS parsing order configuration incomplete_stream")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--model", choices=ALLOWED, default=ALLOWED[0])
    ap.add_argument("--warm-repeats", type=int, default=7)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-available-mib", type=int, default=4608)
    ap.add_argument("--expected-uid", type=int)
    ap.add_argument("--gguf-sha256")
    ap.add_argument("--output")
    args = ap.parse_args(argv)
    if args.self_test:
        selftest()
        return 0
    if not 5 <= args.warm_repeats <= 30 or not 1 <= args.blocks <= 10:
        ap.error("warm-repeats must be 5..30 and blocks 1..10")
    if args.min_available_mib < 4096:
        ap.error("minimum available memory may not be less than 4096 MiB")
    order = plan_order(args.blocks, args.seed)
    summary = {"status": "DRY_RUN" if not args.execute else "EXECUTING",
               "model": args.model, "warm_repeats": args.warm_repeats,
               "block_order": order, "min_available_mib": args.min_available_mib,
               "template_mode": "raw_prompts",
               "native_note": "Verify router's --threads=4 --models-max=1 independently."}
    if not args.execute:
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.expected_uid is None or os.getuid() != args.expected_uid:
        ap.error("explicit current host owner UID is required")
    if not args.gguf_sha256 or not re.match(r"^[0-9a-f]{64}$", args.gguf_sha256):
        ap.error("exact GGUF SHA-256 is required")
    if not args.output or not os.path.isabs(args.output):
        ap.error("exclusive absolute private receipt path is required")
    require_headroom(args.min_available_mib)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as receipt:
            header = dict(summary, status="STARTED", model_sha256=args.gguf_sha256,
                          uid=os.getuid(), host=platform.node(), python=sys.version.split()[0],
                          recorded_epoch=time.time(), endpoints=["127.0.0.1:11434", "127.0.0.1:18086"])
            receipt.write(json.dumps(header, sort_keys=True) + "\n")
            for engine in order:
                for i in range(args.warm_repeats + 1):
                    before = require_headroom(args.min_available_mib)
                    measured = measure(engine, args.model)
                    measured.update({"phase": "load_or_cold" if i == 0 else "warm",
                                     "meminfo_before_kib": before,
                                     "block_engine": engine})
                    receipt.write(json.dumps(measured, sort_keys=True) + "\n")
                    receipt.flush()
                    os.fsync(receipt.fileno())
            receipt.write(json.dumps({"status": "DONE"}) + "\n")
            receipt.flush()
            os.fsync(receipt.fileno())
    except Exception:
        print("RUN_INTERRUPTED: inspect exclusive receipt; no automatic model unload or restart.",
              file=sys.stderr)
        raise
    print("COMPLETE_PRIVATE_RECEIPT " + args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
