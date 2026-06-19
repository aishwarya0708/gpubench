"""
GPUBench: GPU Performance Profiler & Telemetry Analyzer
Author: Aishwarya Vedaraman
Target: NVIDIA Data Analysis Intern — Applied System Engineering (JR2018687)
"""

import time
import json
import csv
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

try:
    import cupy as cp
    GPU_AVAILABLE = True
    GPU_NAME = "NVIDIA GPU (CuPy)"
    _ = cp.array([1.0])
    cp.cuda.Stream.null.synchronize()
    print("✅  CuPy detected — GPU benchmarks will run.")
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = "CPU-only (CuPy not installed)"
    print("⚠️   CuPy not found — GPU columns will be simulated for demo purposes.")

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_NAME = pynvml.nvmlDeviceGetName(handle)
    if isinstance(GPU_NAME, bytes):
        GPU_NAME = GPU_NAME.decode()
    print(f"✅  NVML detected — real GPU utilization telemetry active ({GPU_NAME}).")
except Exception:
    NVML_AVAILABLE = False
    print("⚠️   pynvml not found — GPU utilization will be estimated.")

MATRIX_SIZES = [1000, 2000, 4000, 8000]
WARMUP_RUNS  = 2
BENCH_RUNS   = 5
OUTPUT_DIR   = "gpubench_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_gpu_utilization():
    if NVML_AVAILABLE:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(util.gpu)
        except Exception:
            return 0.0
    return 0.0

def get_gpu_memory_used_gb():
    if NVML_AVAILABLE:
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem.used / (1024**3)
        except Exception:
            return 0.0
    return 0.0

def cpu_matmul(size):
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    for _ in range(WARMUP_RUNS):
        _ = np.matmul(A, B)
    times = []
    for _ in range(BENCH_RUNS):
        t0 = time.perf_counter()
        C = np.matmul(A, B)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.std(times))

def gpu_matmul(size):
    if not GPU_AVAILABLE:
        return None, None
    A = cp.random.rand(size, size, dtype=cp.float32)
    B = cp.random.rand(size, size, dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    for _ in range(WARMUP_RUNS):
        _ = cp.matmul(A, B)
        cp.cuda.Stream.null.synchronize()
    times = []
    for _ in range(BENCH_RUNS):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        C = cp.matmul(A, B)
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(np.std(times))

def compute_throughput_gflops(size, elapsed_sec):
    flops = 2.0 * (size ** 3)
    return (flops / elapsed_sec) / 1e9

def run_benchmarks():
    print("\n" + "="*60)
    print("  GPUBench — Matrix Multiplication Telemetry")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    records = []
    for size in MATRIX_SIZES:
        print(f"\n  ▶  Size {size}×{size} ...")
        cpu_lat, cpu_std = cpu_matmul(size)
        cpu_tput = compute_throughput_gflops(size, cpu_lat)
        gpu_util_before = get_gpu_utilization()
        if GPU_AVAILABLE:
            gpu_lat, gpu_std = gpu_matmul(size)
            gpu_util_after  = get_gpu_utilization()
            speedup         = cpu_lat / gpu_lat if gpu_lat else None
            gpu_mem_gb      = get_gpu_memory_used_gb()
            gpu_tput        = compute_throughput_gflops(size, gpu_lat)
        else:
            sim_factor = 8 + (size / 1000) * 4
            gpu_lat    = cpu_lat / sim_factor
            gpu_std    = gpu_lat * 0.03
            gpu_tput   = compute_throughput_gflops(size, gpu_lat)
            speedup    = sim_factor
            gpu_util_after = min(95, 40 + size / 100)
            gpu_mem_gb = size * size * 4 / (1024**3) * 2
        record = {
            "timestamp":        datetime.now().isoformat(),
            "matrix_size":      size,
            "cpu_latency_s":    round(cpu_lat, 6),
            "cpu_latency_std":  round(cpu_std, 6),
            "cpu_throughput_gflops": round(cpu_tput, 2),
            "gpu_latency_s":    round(gpu_lat, 6),
            "gpu_latency_std":  round(gpu_std, 6),
            "gpu_throughput_gflops": round(gpu_tput, 2),
            "speedup_x":        round(speedup, 2) if speedup else None,
            "gpu_utilization_pct": round(gpu_util_after, 1),
            "gpu_memory_used_gb":  round(gpu_mem_gb, 3),
            "gpu_simulated":    not GPU_AVAILABLE,
        }
        records.append(record)
        print(f"     CPU  latency : {cpu_lat*1000:.2f} ms  |  {cpu_tput:.1f} GFLOP/s")
        print(f"     GPU  latency : {gpu_lat*1000:.2f} ms  |  {gpu_tput:.1f} GFLOP/s")
        print(f"     Speedup      : {speedup:.2f}×")
        print(f"     GPU util     : {gpu_util_after:.1f}%  |  Mem: {gpu_mem_gb:.3f} GB")
    return records

def export_telemetry(records):
    json_path = os.path.join(OUTPUT_DIR, "telemetry.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    csv_path = os.path.join(OUTPUT_DIR, "telemetry.csv")
    keys = records[0].keys()
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    print(f"\n  📁  Telemetry saved → {json_path}, {csv_path}")
    return json_path, csv_path

NVIDIA_GREEN = "#76B900"
NVIDIA_DARK  = "#1a1a2e"
STEEL_BLUE   = "#4A90D9"
WARM_GRAY    = "#2d2d2d"

def _style_ax(ax):
    ax.set_facecolor(WARM_GRAY)
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.spines[:].set_color("#555")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("#ccc")

def plot_results(records):
    sizes    = [r["matrix_size"]          for r in records]
    cpu_lat  = [r["cpu_latency_s"] * 1000 for r in records]
    gpu_lat  = [r["gpu_latency_s"] * 1000 for r in records]
    cpu_tput = [r["cpu_throughput_gflops"] for r in records]
    gpu_tput = [r["gpu_throughput_gflops"] for r in records]
    speedups = [r["speedup_x"]            for r in records]
    gpu_util = [r["gpu_utilization_pct"]  for r in records]

    fig = plt.figure(figsize=(16, 10), facecolor=NVIDIA_DARK)
    fig.suptitle("GPUBench — NVIDIA GPU Telemetry Dashboard",
                 color="white", fontsize=16, fontweight="bold", y=0.97)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(sizes)); w = 0.35
    ax1.bar(x - w/2, cpu_lat, w, label="CPU (NumPy)", color=STEEL_BLUE, alpha=0.9)
    ax1.bar(x + w/2, gpu_lat, w, label="GPU (CuPy)",  color=NVIDIA_GREEN, alpha=0.9)
    ax1.set_title("Latency: CPU vs GPU (ms)", color="white", fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels([f"{s//1000}K" for s in sizes])
    ax1.set_xlabel("Matrix Size", color="#aaa"); ax1.set_ylabel("Latency (ms)", color="#aaa")
    ax1.legend(fontsize=8); _style_ax(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(sizes, cpu_tput, "o--", color=STEEL_BLUE,   linewidth=2, markersize=7, label="CPU")
    ax2.plot(sizes, gpu_tput, "s-",  color=NVIDIA_GREEN, linewidth=2, markersize=7, label="GPU")
    ax2.fill_between(sizes, cpu_tput, gpu_tput, alpha=0.1, color=NVIDIA_GREEN)
    ax2.set_title("Throughput (GFLOP/s)", color="white", fontsize=11)
    ax2.set_xlabel("Matrix Size", color="#aaa"); ax2.set_ylabel("GFLOP/s", color="#aaa")
    ax2.legend(fontsize=8); _style_ax(ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    colors = [NVIDIA_GREEN if s >= 10 else "#f0a500" for s in speedups]
    bars   = ax3.bar([f"{s//1000}K" for s in sizes], speedups, color=colors, alpha=0.9)
    for bar, val in zip(bars, speedups):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1f}×", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
    ax3.set_title("GPU Speedup over CPU", color="white", fontsize=11)
    ax3.set_xlabel("Matrix Size", color="#aaa"); ax3.set_ylabel("Speedup (×)", color="#aaa")
    ax3.axhline(y=1, color="#888", linestyle="--", linewidth=1)
    _style_ax(ax3)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(sizes, gpu_util, "D-", color=NVIDIA_GREEN, linewidth=2, markersize=8)
    ax4.fill_between(sizes, 0, gpu_util, alpha=0.2, color=NVIDIA_GREEN)
    ax4.set_ylim(0, 105)
    ax4.set_title("GPU Utilization (%)", color="white", fontsize=11)
    ax4.set_xlabel("Matrix Size", color="#aaa"); ax4.set_ylabel("GPU Util %", color="#aaa")
    _style_ax(ax4)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.loglog(sizes, cpu_lat, "o--", color=STEEL_BLUE,   linewidth=2, markersize=7, label="CPU")
    ax5.loglog(sizes, gpu_lat, "s-",  color=NVIDIA_GREEN, linewidth=2, markersize=7, label="GPU")
    ax5.set_title("Latency Scaling (log-log)", color="white", fontsize=11)
    ax5.set_xlabel("Matrix Size", color="#aaa"); ax5.set_ylabel("Latency (ms)", color="#aaa")
    ax5.legend(fontsize=8); _style_ax(ax5)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    simulated_note = "* Simulated" if records[0]["gpu_simulated"] else ""
    table_data = [["Size", "CPU ms", "GPU ms", "Speedup"]] + \
                 [[f"{r['matrix_size']//1000}K",
                   f"{r['cpu_latency_s']*1000:.1f}",
                   f"{r['gpu_latency_s']*1000:.1f}",
                   f"{r['speedup_x']:.1f}×"] for r in records]
    tbl = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc="center", loc="center", bbox=[0, 0.1, 1, 0.85])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#1e1e2e" if row == 0 else "#252535")
        cell.set_text_props(color="white" if row > 0 else NVIDIA_GREEN,
                            fontweight="bold" if row == 0 else "normal")
        cell.set_edgecolor("#444")
    ax6.set_title(f"Summary{' ' + simulated_note if simulated_note else ''}",
                  color="white", fontsize=11)

    plot_path = os.path.join(OUTPUT_DIR, "gpubench_dashboard.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=NVIDIA_DARK)
    plt.close()
    print(f"  📊  Dashboard saved → {plot_path}")
    return plot_path

def classify_workloads(records):
    print("\n" + "="*60)
    print("  Workload Classification — Random Forest on Telemetry")
    print("="*60)
    expanded = []
    labels   = []
    rng = np.random.default_rng(42)
    workload_profiles = {
        "memory_bound":  {"util_range": (20, 55),  "tput_ratio": 0.5},
        "compute_bound": {"util_range": (75, 98),  "tput_ratio": 1.8},
        "balanced":      {"util_range": (55, 80),  "tput_ratio": 1.1},
        "cpu_dominated": {"util_range": (5,  20),  "tput_ratio": 0.2},
    }
    for _ in range(60):
        for label, profile in workload_profiles.items():
            for r in records:
                gpu_util = rng.uniform(*profile["util_range"])
                gpu_tput = r["gpu_throughput_gflops"] * profile["tput_ratio"] * rng.uniform(0.9, 1.1)
                speedup  = r["speedup_x"] * profile["tput_ratio"] * rng.uniform(0.85, 1.15)
                cpu_tput = r["cpu_throughput_gflops"] * rng.uniform(0.92, 1.08)
                expanded.append([r["matrix_size"], r["cpu_latency_s"]*1000,
                                  r["gpu_latency_s"]*1000, gpu_util, gpu_tput,
                                  cpu_tput, speedup, r["gpu_memory_used_gb"]])
                labels.append(label)
    X = np.array(expanded)
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    print(f"\n  Accuracy: {acc*100:.1f}%")
    print(f"\n  {classification_report(y_test, y_pred, target_names=le.classes_)}")

    feat_names  = ["MatrixSize","CPU_Lat","GPU_Lat","GPU_Util%","GPU_GFLOPS","CPU_GFLOPS","Speedup","GPU_Mem"]
    importances = clf.feature_importances_
    order       = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=NVIDIA_DARK)
    ax.set_facecolor(WARM_GRAY)
    bars = ax.bar(range(len(feat_names)), importances[order], color=NVIDIA_GREEN, alpha=0.9)
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels([feat_names[i] for i in order], rotation=35, ha="right", color="#ccc", fontsize=9)
    ax.set_title("Feature Importances — Workload Classifier", color="white", fontsize=12, fontweight="bold")
    ax.set_ylabel("Importance", color="#aaa")
    ax.tick_params(colors="#aaa"); ax.spines[:].set_color("#555")
    for bar, imp in zip(bars, importances[order]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{imp:.3f}", ha="center", va="bottom", color="white", fontsize=8)
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(fi_path, dpi=150, bbox_inches="tight", facecolor=NVIDIA_DARK)
    plt.close()
    print(f"  📊  Feature importance chart → {fi_path}")

    print("\n  🔍  Classifying live benchmark runs:")
    for r in records:
        features = np.array([[r["matrix_size"], r["cpu_latency_s"]*1000,
                               r["gpu_latency_s"]*1000, r["gpu_utilization_pct"],
                               r["gpu_throughput_gflops"], r["cpu_throughput_gflops"],
                               r["speedup_x"], r["gpu_memory_used_gb"]]])
        pred_label = le.inverse_transform(clf.predict(features))[0]
        confidence = clf.predict_proba(features)[0].max() * 100
        print(f"     {r['matrix_size']//1000}K matrix → {pred_label:>18}  (confidence: {confidence:.1f}%)")

    result_path = os.path.join(OUTPUT_DIR, "classifier_report.json")
    serializable = {}
    for k, v in report.items():
        if isinstance(v, dict):
            serializable[k] = {m: round(val, 4) if isinstance(val, float) else val for m, val in v.items()}
        else:
            serializable[k] = round(v, 4) if isinstance(v, float) else v
    with open(result_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  📁  Classifier report → {result_path}")
    return acc, report

def write_results_table(records):
    lines = [
        "## Benchmark Results\n",
        "| Matrix Size | CPU Latency (ms) | GPU Latency (ms) | CPU GFLOP/s | GPU GFLOP/s | Speedup | GPU Util % |",
        "|-------------|-----------------|-----------------|-------------|-------------|---------|------------|",
    ]
    for r in records:
        lines.append(
            f"| {r['matrix_size']:>11} "
            f"| {r['cpu_latency_s']*1000:>15.2f} "
            f"| {r['gpu_latency_s']*1000:>15.2f} "
            f"| {r['cpu_throughput_gflops']:>11.1f} "
            f"| {r['gpu_throughput_gflops']:>11.1f} "
            f"| {r['speedup_x']:>6.2f}× "
            f"| {r['gpu_utilization_pct']:>10.1f} |"
        )
    table_path = os.path.join(OUTPUT_DIR, "results_table.md")
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  📝  Results table → {table_path}")
    return "\n".join(lines)

if __name__ == "__main__":
    records = run_benchmarks()
    json_path, csv_path = export_telemetry(records)
    plot_path = plot_results(records)
    acc, _    = classify_workloads(records)
    table     = write_results_table(records)
    print("\n" + "="*60)
    print("  ✅  GPUBench complete!")
    print(f"  Output directory : ./gpubench_output/")
    print(f"  Dashboard        : gpubench_dashboard.png")
    print(f"  Feature chart    : feature_importance.png")
    print(f"  Telemetry        : telemetry.json / telemetry.csv")
    print(f"  Classifier acc   : {acc*100:.1f}%")
    print("="*60)
