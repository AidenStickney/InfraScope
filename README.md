# InfraScope: Profiling Cross-Layer Hardware Interactions in ML Training & Inference Workloads

This project explores whether hardware telemetry from ML training and inference workloads can reveal cross-layer performance bottlenecks through frequent pattern mining. The core idea is to turn raw, high-frequency sensor data (CPU utilization, GPU power, memory pressure, network I/O, etc.) into symbolic representations that capture the system state at each moment, and then mine those representations for recurring patterns and association rules.

## What This Is

A course project for **CSCE 676: Data Mining & Analysis** (Spring 2026, Texas A&M University).

The dataset is [Reveal CPU & GPU Telemetry Dataset](https://huggingface.co/datasets/subsetchen/RevealTelemetryDatasetforMLInfraProfilingAnomalyDetection). Roughly 100 GB of 100 ms sampled hardware telemetry collected from HPC nodes running diverse ML workloads (training & inference for models like DeepSeek, BERT, ResNet, etc.). Each file is a time-series with 700–800 columns spanning GPU utilization, CPU performance counters, memory stats, and network traffic.

## Project Structure

```
CSCE676/
├── README.md                    # here
├── 731002406_checkpoint1.ipynb  # main EDA notebook (Checkpoint 1)
├── setup.sh                     # module loads for the TAMU HPRC cluster
└── data/
    ├── meta_gpu.csv             # metadata for all GPU telemetry files
    ├── meta_cpu.csv             # metadata for all CPU telemetry files
    ├── Reveal_GPU/
    │   ├── train_separated/     # ~740 GPU training run files
    │   └── infer_separated/     # ~740 GPU inference run files
    └── Reveal_CPU/
        ├── train_separated/     # ~1700 CPU training run files
        └── infer_separated/     # ~1340 CPU inference run files
```

## How It Works

The EDA notebook (`731002406_checkpoint1.ipynb`) walks through the full pipeline:

1. **Data inventory**: Loads the meta CSVs to understand the scope (4500+ runs across GPU and CPU hosts), checks schema consistency, and profiles workload distributions.

2. **Quality assessment**: Samples 20 files from each hardware type and flags problematic metrics: always-missing columns, structural zeros (e.g., network error counters that are forever 0), and near-constant system values (e.g., `MemTotal`). About 800 GPU metrics and 1,500 CPU metrics get excluded.

3. **Curated metric selection**: Picks 11 metric groups for GPU (GPU utilization, memory utilization, power draw, temperature, CPU busy %, IPC, stall ratio, memory utilization, memory available, network rx/tx) and 6-7 for CPU (same minus GPU-specific ones). Metrics matched using regex patterns so that it works across different hostnames.

4. **Windowing & discretization**: Each run is sliced into 1 second windows (10 rows). Within each window, the median of each metric group is computed and discretized into LOW/MED/HIGH using global tercile thresholds. This produces one symbolic basket per window.

5. **Basket EDA** — Analyzes the resulting baskets: size distributions, item frequency, co-occurrence heatmaps, per-run basket counts, and a feasibility check at various support thresholds to make sure Apriori & FP-growth will actually be tractable.

6. **Window sensitivity** — Repeats the basket analysis at 2s and 5s window sizes to show the results are stable across reasonable choices.

## Running It

This was run on the [TAMU HPRC Grace cluster](https://hprc.tamu.edu/wiki/Grace:Intro). The data is large (~100 GB unzipped) so it needs to run on a machine with enough disk and memory.

```bash
# On the cluster:
module load GCCcore/14.2.0
module load Python/3.13.1

# Set up a venv (first time only)
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn

# Run the notebook
jupyter notebook 731002406_checkpoint1.ipynb
```

The notebook will download the dataset from HuggingFace automatically if it's not already present.

## Key Findings (Checkpoint 1)

- GPU telemetry produces ~4,400 baskets from 20 sampled files, with 33 unique items and rich co-occurrence structure. Feasibility looks good for itemset mining at 1% support.
- CPU telemetry produces thousands of baskets once the host-agnostic matching is applied.
- The three-level discretization (LOW/MED/HIGH) with global tercile thresholds is simple but effective. Every basket has the same number of items (equal to the number of curated groups), so the variation is entirely in the content of baskets, not their size.
- Network metrics and memory availability show the widest spread across levels, while GPU temperature is relatively stable (mostly LOW).

## What's Next

- Run Apriori / FP-growth on the constructed baskets to find frequent itemsets and association rules.
- Explore sequential pattern mining to capture temporal transitions (e.g., "GPU utilization HIGH followed by memory pressure HIGH").
- Compare patterns across workload types (training vs. inference, different models).

## References

Chen, Z., Chien, S. W. D., Qian, P., & Zilberman, N. (2025). *Detecting Anomalies in Machine Learning Infrastructure via Hardware Telemetry*. arXiv:2510.26008.

Dataset: [Reveal Telemetry Dataset](https://huggingface.co/datasets/subsetchen/RevealTelemetryDatasetforMLInfraProfilingAnomalyDetection) (CC BY 4.0)
