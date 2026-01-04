# VolFormer: A Transformer-Based Approach to Stock Volatility Forecasting

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/Status-Refactored-success)

## 📌 Project Overview

**VolFormer** is a specialized Transformer architecture designed to forecast **Realized Volatility (RV)** in financial markets. Developed as part of a Master's Thesis at UC3M, this project bridges the gap between state-of-the-art Natural Language Processing techniques and financial time-series analysis.

By treating intraday return sequences as "tokens," VolFormer leverages self-attention mechanisms to capture complex, non-linear temporal dependencies that traditional econometric models (like HAR-RV) often miss.

> **Why Volatility?**
> Accurate volatility forecasting is the cornerstone of **quantitative risk management** and **options pricing**. In a market driven by uncertainty, the ability to anticipate variance allows institutional investors to hedge effectively and price derivatives with greater precision.

---

## 🚀 Key Engineering Features

This repository represents a **production-grade refactoring** of the original research code, emphasizing modularity, reproducibility, and scalability.

- **🏗️ Modular Architecture**: The monolithic research code has been decoupled into distinct domains (`data`, `models`, `training`), ensuring strict separation of concerns and testability.
- **⚙️ Config-Driven Experiments**: Training runs are fully reproducible via centralized YAML configuration files. No more hardcoded hyperparameters.
- **📊 Professional Logging**: Integrated `logging` module replaces ad-hoc print statements, providing structured, readable logs for experiment tracking.
- **⚡ CUDA-Optimized**: Data loaders and training loops are optimized for GPU acceleration, utilizing Mixed Precision (AMP) and pinned memory for maximum throughput.
- **🛡️ Data Leakage Protection**: The preprocessing pipeline implements rigorous checks (e.g., removing overnight returns, strict day alignment) to ensure no future information leaks into the training set—a critical requirement for valid financial backtesting.

---

## 🧠 The VolFormer Model

At its core, VolFormer adapts the standard Transformer Encoder for regression tasks:
*   **Input**: Rolling window of intraday log-returns (e.g., 10-minute intervals).
*   **Positional Encodings**: Learnable embeddings to retain temporal order.
*   **Self-Attention**: Multi-head attention layers that allow the model to weigh the importance of different intraday periods dynamically.
*   **CLS Token**: A learnable token (similar to BERT) pools the sequence information into a single context vector for the final volatility prediction.

---

## 📂 Project Structure

```text
.
├── configs/                 # Experiment configurations
│   ├── default.yaml         # Main production config
│   └── test_config.yaml     # Fast CI/CD test config
├── plots/                   # Visualization assets
├── raw_data/                # Dataset storage
├── src/                     # Source code
│   ├── data/                # Data ingestion & preprocessing
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/              # PyTorch model definitions
│   │   └── volformer.py
│   └── training/            # Training loop & utilities
│       ├── trainer.py
│       └── logger.py
├── train.py                 # CLI entry point
└── requirements.txt         # Dependencies
```

---

## 🛠️ Getting Started

### Prerequisites
*   Python 3.9+
*   CUDA-capable GPU (recommended)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Start-sys/volformer.git
    cd volformer
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running Experiments

To start a training session using the default configuration:

```bash
python3 train.py --config configs/default.yaml
```

To run a custom experiment, duplicate a config file, modify the parameters (e.g., `d_model`, `lr`), and run:

```bash
python3 train.py --config configs/my_experiment.yaml
```

---

## 📈 Visuals & Results

One of the key advantages of Attention mechanisms is **interpretability**. The heatmaps below illustrate how VolFormer attends to specific intraday intervals during high-volatility events, effectively "learning" which times of day are most predictive.

| **Head 1 Attention Pattern** | **Head 2 Attention Pattern** |
|:----------------------------:|:----------------------------:|
| ![Head 1](plots/attention_Apple_L2_H1_day349.png) | ![Head 2](plots/attention_Apple_L2_H2_day349.png) |
| *Focus on specific market open intervals* | *Broader attention across the trading day* |

---

## 🎓 Academic Context

This project was originally developed for a **Master's Thesis** at **Universidad Carlos III de Madrid (UC3M)**. The research focused on benchmarking Deep Learning architectures against established heterogeneous autoregressive (HAR) models in the context of high-frequency trading data.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
