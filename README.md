# SPARK: Test Case Prioritization for Deep Learning Framework Testing

This repository is the open artifact of the paper **"Test Case Prioritization for Deep Learning Framework Testing"**.

## Description

In this work, we propose SPARK, a test case prioritization framework for DL framework testing. SPARK represents each generated test case as a computational graph, estimates its pre-execution utility with six hierarchical graph-based coverage metrics spanning the operator, subgraph, and global levels, captures structural redundancy through Weisfeiler-Lehman subtree features, SimHash, and locality-sensitive hashing, and then uses an exploration-first adaptive scheduler with lightweight bug signatures to move bug-revealing test cases earlier in execution. Besides, SPARK substantially improves prioritization effectiveness and early bug detection across nine testing tools on seven DL frameworks and backends.

If you have any questions, please leave a message here to contact us.

## Included Upstream Testing Tools

The benchmark in the paper is built on the following nine open-source testing tools:

- COMET
- DevMuT
- Gandalf
- GenCoG
- LEMON
- ModelMeta
- Muffin
- NEURI
- NNSmith

The local copies under `tools/` are included for artifact integration and experiment reproduction. See [`tools/README.md`](tools/README.md) for the upstream repository mapping.

## Benchmark

The paper evaluates SPARK on:

- 9 testing tools
- 7 frameworks/backends: PyTorch, TensorFlow, MindSpore, ONNX Runtime, TVM, TFLite, and TensorRT
- 450 benchmark test suites
- more than 40,000 generated test cases

For each tool, each run generates 100 test cases and the generation process is repeated 50 times.

## Results

The current repository includes the result plots used in the paper:

- `results/empirical_results/rq1.pdf` — empirical RQ1 results: APFD and Top-20% budget bug ratios for lightweight execution orders (Fig. 2).
- `results/empirical_results/rq2.pdf` — empirical RQ2 results: APFD, Top-20% bugs, first-bug rank, and Spearman correlation for lightweight criteria (Fig. 3).
- `results/main_results/rq3/apfd.pdf` — APFD distributions for SPARK and the internal baselines across the nine testing tools (Fig. 6).
- `results/main_results/rq3/apfdc.pdf` — APFDc distributions for SPARK and the internal baselines across the nine testing tools (Fig. 7).
- `results/main_results/rq4/curve.pdf` — cumulative bug-detection progress curves (Fig. 8).
- `results/main_results/rq5/coverage.pdf` — OCS progression curves during execution (Fig. 9).
- `results/main_results/rq6/complexity_total.pdf` — total prioritization runtime by method (Fig. 10a).
- `results/main_results/rq6/complexity_core.pdf` — SPARK runtime decomposition (Fig. 10b).

## Setup

### Minimal setup for the SPARK core

```bash
python -m venv .venv
source .venv/bin/activate
export PYTHONPATH=src
```

### Experimental environment

- Python 3.10.20
- PyTorch 2.5.1
- NumPy 1.26.4
- pandas 2.3.3
- NetworkX 3.4.2
- Ubuntu 24.04.1 LTS
- NVIDIA GeForce RTX 4090 (24GB), CUDA 12.4

If you want to reproduce the full experiments from the paper, you will also need the corresponding environments and dependencies of the upstream tools under `tools/`.
