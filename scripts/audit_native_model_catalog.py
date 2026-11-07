#!/usr/bin/env python3
"""Read-only, fail-closed Ollama/native-router model-weight identity audit.

Does not load, generate, unload, copy or modify models. A weight-blob match
alone never establishes Modelfile, tokenization, tool or API parity.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import stat
import sys
from urllib.parse import urlparse
from urllib.request import build_opener, HTTPRedirectHandler, ProxyHandler

BLOB = re.compile(r"^(?:sha256:|sha256-)([0-9a-f]{64})$")
MODEL_LAYER = "application/vnd.ollama.image.model"


class RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def sha(value):
    match = BLOB.fullmatch(value) if isinstance(value, str) else None
    return match.group(1) if match else None


def manifest_model_blob(data):
    layers = data.get("layers") if isinstance(data, dict) else None
    if not isinstance(layers, list):
        return None
    models = [layer for layer in layers
              if isinstance(layer, dict) and layer.get("mediaType") == MODEL_LAYER]
    return sha(models[0].get("digest")) if len(models) == 1 else None


def regular_file(path):
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except (OSError, ValueError):
        return False


def list_manifests(root):
    manifests = root / "manifests"
    result = {}
    if not manifests.is_dir() or manifests.is_symlink():
        return result
    for parent, directories, files in os.walk(manifests, followlinks=False):
        directories[:] = [d for d in directories if not (Path(parent) / d).is_symlink()]
        for name in files:
            file_path = Path(parent) / name
            if not regular_file(file_path):
                continue
            rel = file_path.relative_to(manifests).parts
            if len(rel) != 4:
                continue
            registry, namespace, model, tag = rel
            if registry.lower() != "registry.ollama.ai":
                alias = registry + "/" + namespace + "/" + model + ":" + tag
            elif namespace.lower() != "library":
                alias = namespace + "/" + model + ":" + tag
            else:
                alias = model + ":" + tag
            try:
                with file_path.open(encoding="utf-8") as stream:
                    digest = manifest_model_blob(json.load(stream))
            except (OSError, ValueError, UnicodeError):
                digest = None
            blob = root / "blobs" / ("sha256-" + digest) if digest else None
            result[alias] = digest if blob and not (root / "blobs").is_symlink() and regular_file(blob) else None
    return result


def native_model_blob(item, root):
    status = item.get("status") if isinstance(item, dict) else None
    argv = status.get("args") if isinstance(status, dict) else None
    if not isinstance(argv, list) or argv.count("--model") != 1:
        return None, "unresolved_native_model_path"
    idx = argv.index("--model")
    if idx + 1 >= len(argv) or not isinstance(argv[idx + 1], str):
        return None, "unresolved_native_model_path"
    path = Path(argv[idx + 1])
    digest = sha(path.name)
    expected = root / "blobs" / ("sha256-" + digest) if digest else None
    if not expected or path != expected or (root / "blobs").is_symlink():
        return None, "noncanonical_native_model_path"
    if not regular_file(path):
        return None, "missing_native_blob"
    return digest, None


def reconcile(ollama_catalog, native_catalog, manifests, root):
    if not isinstance(ollama_catalog, dict) or not isinstance(ollama_catalog.get("models"), list):
        raise ValueError("Invalid Ollama tags catalog")
    if not isinstance(native_catalog, dict) or not isinstance(native_catalog.get("data"), list):
        raise ValueError("Invalid native models catalog")
    by_digest, ollama_rows = {}, []
    for item in ollama_catalog["models"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("Invalid Ollama catalog entry")
        alias = item["name"]
        digest = manifests.get(alias)
        ollama_rows.append({"alias": alias, "model_blob_sha256": digest,
                            "identity": "manifest_blob_verified" if digest else "unresolved_manifest_or_blob"})
        if digest:
            by_digest.setdefault(digest, []).append(alias)
    native_rows = []
    for item in native_catalog["data"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("Invalid native catalog entry")
        digest, problem = native_model_blob(item, root)
        matches = sorted(by_digest.get(digest, [])) if digest else []
        status = problem or ("shared_model_weights" if matches else "native_blob_without_ollama_alias")
        native_rows.append({"alias": item["id"], "model_blob_sha256": digest,
                            "ollama_aliases_with_identical_weights": matches, "identity": status})
    native_digests = {row["model_blob_sha256"] for row in native_rows if row["model_blob_sha256"]}
    for row in ollama_rows:
        if row["model_blob_sha256"] and row["model_blob_sha256"] not in native_digests:
            row["identity"] = "ollama_weights_not_in_native_catalog"
    return {
        "summary": {
            "ollama_alias_count": len(ollama_rows), "native_alias_count": len(native_rows),
            "native_aliases_matching_ollama_weights": sum(x["identity"] == "shared_model_weights" for x in native_rows),
            "native_weights_not_in_ollama_catalog": sum(x["identity"] == "native_blob_without_ollama_alias" for x in native_rows),
            "ollama_aliases_missing_native_weight": sum(x["identity"] == "ollama_weights_not_in_native_catalog" for x in ollama_rows),
            "unresolved_ollama": sum(x["identity"] == "unresolved_manifest_or_blob" for x in ollama_rows),
            "unresolved_native": sum(x["identity"] in ("unresolved_native_model_path", "noncanonical_native_model_path", "missing_native_blob") for x in native_rows),
            "missing_native_blobs": sum(x["identity"] == "missing_native_blob" for x in native_rows),
        },
        "ollama": sorted(ollama_rows, key=lambda x: x["alias"]),
        "native": sorted(native_rows, key=lambda x: x["alias"]),
    }


def fetch_local(base, path):
    parsed = urlparse(base)
    if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Only loopback catalog endpoints are authorised")
    opener = build_opener(ProxyHandler({}), RejectRedirect())
    with opener.open(base.rstrip("/") + path, timeout=8) as response:
        if response.status != 200:
            raise RuntimeError("Catalog endpoint returned non-200")
        return json.load(response)


def selftest(test_root):
    from tempfile import TemporaryDirectory
    if not test_root.is_absolute() or not test_root.is_dir() or test_root.is_symlink():
        raise ValueError("Self-test requires an existing explicit owner directory, never /tmp")
    with TemporaryDirectory(dir=str(test_root)) as folder:
        root = Path(folder)
        digest = "a" * 64
        (root / "blobs").mkdir()
        (root / "blobs" / ("sha256-" + digest)).write_bytes(b"fixture")
        manifest = root / "manifests" / "registry.ollama.ai" / "library" / "tiny" / "latest"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"layers": [{"mediaType": MODEL_LAYER, "digest": "sha256:" + digest}]}))
        other_registry = root / "manifests" / "mirror.example.net" / "library" / "tiny" / "latest"
        other_registry.parent.mkdir(parents=True)
        other_registry.write_text(json.dumps({"layers": [{"mediaType": MODEL_LAYER, "digest": "sha256:" + "b" * 64}]}))
        (root / "blobs" / ("sha256-" + "b" * 64)).write_bytes(b"other registry fixture")
        assert list_manifests(root)["mirror.example.net/library/tiny:latest"] == "b" * 64
        assert list_manifests(root)["tiny:latest"] == digest
        tags = {"models": [{"name": "tiny:latest"}, {"name": "unresolved:latest"}]}
        models = {"data": [
            {"id": "TINY", "status": {"args": ["--model", str(root / "blobs" / ("sha256-" + digest))]}},
            {"id": "OTHER_HOST", "status": {"args": ["--model", "/another/root/blobs/sha256-" + digest]}},
            {"id": "INVALID", "status": {"args": ["--model", str(root / "blobs" / ("sha256-" + "c" * 64))]}},
        ]}
        result = reconcile(tags, models, list_manifests(root), root)
        assert result["summary"] == {
            "ollama_alias_count": 2, "native_alias_count": 3,
            "native_aliases_matching_ollama_weights": 1,
            "native_weights_not_in_ollama_catalog": 0,
            "ollama_aliases_missing_native_weight": 0,
            "unresolved_ollama": 1, "unresolved_native": 2, "missing_native_blobs": 1,
        }
        assert next(x for x in result["native"] if x["alias"] == "TINY")["ollama_aliases_with_identical_weights"] == ["tiny:latest"]
        assert RejectRedirect().redirect_request(None, None, 302, "redirect", {}, "http://external.invalid/") is None
        # A symlinked blob root is never admissible, even when its final file is regular.
        linked_root = root / "other_root"
        linked_root.mkdir()
        try:
            (linked_root / "blobs").symlink_to(root / "blobs", target_is_directory=True)
        except (NotImplementedError, OSError):
            pass
        else:
            linked_alias = {"status": {"args": ["--model", str(linked_root / "blobs" / ("sha256-" + digest))]}}
            assert native_model_blob(linked_alias, linked_root) == (None, "noncanonical_native_model_path")
        assert manifest_model_blob({"layers": [
            {"mediaType": MODEL_LAYER, "digest": "sha256:" + digest},
            {"mediaType": MODEL_LAYER, "digest": "sha256:" + "b" * 64},
        ]}) is None
    print("SELFTEST_PASS exact_blob_catalog_identity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--ollama-base", default="http://127.0.0.1:11434")
    parser.add_argument("--native-base", default="http://127.0.0.1:18086")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if not args.self_test_root:
            parser.error("--self-test-root is required; /tmp staging is not permitted")
        selftest(args.self_test_root)
        return
    if not args.models_root or not args.models_root.is_absolute():
        parser.error("--models-root must be the explicit absolute owner models directory")
    root = args.models_root
    if not root.is_dir() or root.is_symlink():
        parser.error("Models root must be an existing non-symlink directory")
    result = reconcile(fetch_local(args.ollama_base, "/api/tags"),
                       fetch_local(args.native_base, "/models"), list_manifests(root), root)
    result["scope"] = "MODEL_WEIGHT_IDENTITY_ONLY_NOT_FUNCTIONAL_PARITY"
    result["observed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        sys.exit(1)
