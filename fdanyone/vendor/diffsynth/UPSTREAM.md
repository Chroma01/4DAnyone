# DiffSynth-Studio provenance

This directory is a deliberately small inference-only extract from [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), licensed under Apache-2.0.

- Public base revision: `04e39f7de53df7276a7b40ca1791c2a393e05ff3`
- Research fork revision used by the original experiment: `c00782d90c872c97bda4745a9e6a41a0a4a7c4db`
- `UPSTREAM.patch` SHA-256: `178e6035e451f94a2122fa4d2c876a488546964768b90275549cbf609f97daba`
- Extracted: 2026-07-16

The research fork revision was not anonymously reachable when the release contract was audited. `UPSTREAM.patch` therefore records the exact binary-safe diff from the public base to the research revision for the retained Wan/SpaTem/scheduler source files. Unrelated research-fork changes are deliberately excluded. `VENDORED_FILES.txt` is the reviewable extraction manifest.

4DAnyone subsequently removed registry, downloader, training, image-encoder, and unrelated pipeline surfaces; changed imports to the local vendored package; exposes direct, typed model construction in first-party code; and exposes the implementation selected by the retained automatic attention fallback chain for run metadata. These pruning and observability changes do not alter the retained Wan/SpaTem architecture or scheduler math.

To reproduce the retained research sources before pruning:

```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
git -C DiffSynth-Studio checkout 04e39f7de53df7276a7b40ca1791c2a393e05ff3
git -C DiffSynth-Studio apply --check /path/to/UPSTREAM.patch
git -C DiffSynth-Studio apply /path/to/UPSTREAM.patch
```

The patch contains research-code additions and is itself source code. It must remain covered by this directory's Apache-2.0 `LICENSE` and attribution.
