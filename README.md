# Full-Stack AI Developer Program — Phase 01: Day 5
## Data Processing with NumPy: Vectorized Micro-Tools & Anomaly Detection

This repository contains practical implementations of vectorized computing tasks using Python and NumPy. The goal of this module is to demonstrate the performance advantages of NumPy’s element-wise array math over traditional Python loop-based calculations.

---

## 📂 Project Architecture & Contents

The project is split into two components: experimental micro-tools (Tasks 1–4) and an end-to-end network security analytic model (Task 5).

### 🛠️ Part 1: Tasks 1–4 (Small Vectorized Tools)
Four mini-scripts demonstrating how core NumPy concepts map directly to real-world business mechanics:
1. **Grocery Bill Calculator:** Computes a full batch order total instantly by multiplying a `prices` matrix array with a `quantities` matrix array using element-wise vector operations and the `.sum()` method.
2. **Temperature Converter Batch:** Applies the algebraic calculation $F = (C \times \frac{9}{5}) + 32$ across an entire week's worth of Celsius numbers in a single execution line.
3. **Exam Curve Applier:** Models flat administrative score-scaling across standard arrays and evaluates the structural shift in mean (`.mean()`) statistics before and after modification.
4. **Weekly Sales Quick Stats:** Demonstrates boolean array indexing (masking) to query, segment, and pull out specific numeric data points matching conditional ranges (e.g., retrieving sales figures above the weekly mean).

---

### 🛡️ Part 2: Task 5 (Main Project: Secure Net)
**Project Title:** Secure Net — Cyber Threat and Anomaly Detection  
**Data Dependency:** `nsl_kdd_dataset.csv` (Derived from the benchmark NSL-KDD Intrusion Detection Framework).

The core analytics script builds an entry-level security engine to read standard production-grade telemetry logs and look for active patterns or systemic behavior anomalies.

#### Core Functionalities Implemented:
* **Fault-Tolerant I/O:** Safely accesses dataset parameters inside a `try/except` environment block to isolate `FileNotFoundError` scenarios.
* **Header Parsing Automation:** Sanitizes column tags programmatically to ignore trailing whitespace errors and maps target labels (`class`/`label`) dynamically.
* **Vectorized Feature Normalization:** Converts multi-dimensional metrics (connection `duration`, `src_bytes`, `dst_bytes`) into scaled $0.1$ distribution ranges using vectorized Min-Max logic:
  $$x_{\text{norm}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$$
* **Threat Metric Tracking:** Leverages boolean masking models to identify anomalies instantly and outputs the network's overarching percentage threat index.
* **Outlier Tracking & Matrix Sorting:** Employs `np.argmax()` to catch severe transmission spikes, and `np.argsort()[::-1]` to sort and rank network connections from longest to shortest duration.
* **Performance Benchmarking:** Runs a real-time execution race against standard Python loops to show how much faster NumPy vectorization handles data at scale.

---

## 🚀 Getting Started & Execution

### Prerequisites
Ensure you have a modern Python 3 instance running along with the `numpy` package:
```bash
pip install numpy
