"""
Memory Scaling Visualization
=============================

Creates three publication-quality plots showing Phase 4A.2 baseline characteristics.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats


def load_baseline_data():
    """Load baseline measurements."""
    summary_path = Path(__file__).parent / "tests" / "memory_profiles" / "baseline_summary.json"
    with open(summary_path) as f:
        data = json.load(f)
    
    sizes = []
    peaks_mean = []
    peaks_std = []
    
    for size_str, metrics in sorted(data['summary'].items()):
        size_mb = float(size_str.replace('MB', ''))
        sizes.append(size_mb)
        peaks_mean.append(metrics['peak_mean_mb'])
        peaks_std.append(metrics['peak_std_mb'])
    
    return np.array(sizes), np.array(peaks_mean), np.array(peaks_std)


def plot_memory_scaling():
    """Create log-log scaling plot with power law fit."""
    sizes, peaks, peaks_std = load_baseline_data()
    
    # Filter out zero sizes for log scale
    valid_idx = sizes > 0.01
    sizes_valid = sizes[valid_idx]
    peaks_valid = peaks[valid_idx]
    peaks_std_valid = peaks_std[valid_idx]
    
    # Fit power law
    log_sizes = np.log(sizes_valid)
    log_peaks = np.log(peaks_valid)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_peaks)
    
    # Generate fit line
    sizes_fit = np.logspace(np.log10(sizes_valid.min()), np.log10(sizes_valid.max()), 100)
    peaks_fit = np.exp(intercept + slope * np.log(sizes_fit))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot data with error bars
    ax.errorbar(sizes_valid, peaks_valid, yerr=peaks_std_valid, 
                fmt='o', markersize=10, capsize=8, capthick=2,
                color='#2E86C1', ecolor='#1B4965', label='Measured peak memory',
                linewidth=2)
    
    # Plot fit line
    ax.loglog(sizes_fit, peaks_fit, 'r--', linewidth=2.5, alpha=0.8,
              label=f'Power law fit: y = Size^{slope:.3f}')
    
    # Confidence bands
    upper_band = peaks_fit * (1 + 2 * std_err)
    lower_band = peaks_fit * (1 - 2 * std_err)
    ax.fill_between(sizes_fit, lower_band, upper_band, alpha=0.15, color='red')
    
    # Reference lines for O(n), O(log n), O(1)
    reference_size = 1.0
    reference_memory = 50.0
    
    # O(1) line
    ax.axhline(y=reference_memory, color='green', linestyle=':', linewidth=2, 
               alpha=0.6, label='O(1) - Constant')
    
    # O(log n) line
    log_line = reference_memory * np.log(sizes_fit) / np.log(reference_size)
    ax.loglog(sizes_fit, log_line, color='blue', linestyle=':', linewidth=2,
              alpha=0.6, label='O(log n) - Logarithmic')
    
    # O(n) line
    ax.loglog(sizes_fit, sizes_fit * (reference_memory / reference_size), 
              color='orange', linestyle=':', linewidth=2, alpha=0.6,
              label='O(n) - Linear')
    
    # Formatting
    ax.set_xlabel('Input Size (MB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Peak Memory Usage (MB)', fontsize=14, fontweight='bold')
    ax.set_title('Phase 4A.2: Memory Scaling Analysis\nSub-Linear Growth: Memory ∝ Size^0.055',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    
    # Add annotation
    ax.text(0.05, 0.95, f'R² = {r_value**2:.4f}\nSlope = {slope:.3f}\nP-value = {p_value:.2e}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent / 'memory_scaling_analysis.png'), dpi=300, bbox_inches='tight')
    print("[OK] Saved: memory_scaling_analysis.png")
    
    return fig


def plot_memory_distribution():
    """Create distribution plot showing variance across sizes."""
    sizes, peaks, peaks_std = load_baseline_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Peak memory by size with confidence intervals
    valid_idx = sizes > 0.01
    sizes_valid = sizes[valid_idx]
    peaks_valid = peaks[valid_idx]
    peaks_std_valid = peaks_std[valid_idx]
    
    x_pos = np.arange(len(sizes_valid))
    
    bars = ax1.bar(x_pos, peaks_valid, color='#3498DB', alpha=0.8, edgecolor='#2C3E50', linewidth=2)
    ax1.errorbar(x_pos, peaks_valid, yerr=1.96*peaks_std_valid, fmt='none', 
                 ecolor='#E74C3C', elinewidth=2, capsize=5, label='95% CI')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, peaks_valid)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}MB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Input Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Peak Memory (MB)', fontsize=12, fontweight='bold')
    ax1.set_title('Peak Memory Usage by Input Size\n(with 95% Confidence Intervals)',
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{s:.1f}MB' for s in sizes_valid], rotation=45)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    # Plot 2: Coefficient of Variation (measurement quality)
    cv_pct = 100 * peaks_std_valid / peaks_valid
    
    colors = ['#27AE60' if cv < 0.5 else '#F39C12' if cv < 1.0 else '#E74C3C' 
              for cv in cv_pct]
    
    bars2 = ax2.bar(x_pos, cv_pct, color=colors, alpha=0.8, edgecolor='#2C3E50', linewidth=2)
    
    # Add threshold lines
    ax2.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, 
                label='Excellent (< 0.5%)')
    ax2.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                label='Good (< 1%)')
    ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label='Acceptable (< 2%)')
    
    ax2.set_xlabel('Input Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Measurement Quality by Input Size\n(Lower is better - indicates reproducibility)',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{s:.1f}MB' for s in sizes_valid], rotation=45)
    ax2.set_ylim(0, max(cv_pct) * 1.1)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent / 'memory_distribution_analysis.png'), dpi=300, bbox_inches='tight')
    print("[OK] Saved: memory_distribution_analysis.png")
    
    return fig


def plot_scaling_efficiency():
    """Create efficiency plot showing memory overhead and compression."""
    sizes, peaks, peaks_std = load_baseline_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Load compression data
    summary_path = Path(__file__).parent / "tests" / "memory_profiles" / "baseline_summary.json"
    with open(summary_path) as f:
        data = json.load(f)
    
    compression_ratios = []
    for size_str, metrics in sorted(data['summary'].items()):
        compression_ratios.append(metrics['compression_ratio'])
    
    compression_ratios = np.array(compression_ratios)
    
    # Filter valid
    valid_idx = sizes > 0.01
    sizes_valid = sizes[valid_idx]
    peaks_valid = peaks[valid_idx]
    compression_valid = compression_ratios[valid_idx]
    
    x_pos = np.arange(len(sizes_valid))
    
    # Plot 1: Memory overhead
    overhead_pct = 100 * (peaks_valid - sizes_valid) / sizes_valid
    
    bars = ax1.bar(x_pos, overhead_pct, color='#E67E22', alpha=0.8, 
                   edgecolor='#2C3E50', linewidth=2)
    
    ax1.set_xlabel('Input Size', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Memory Overhead (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Memory Overhead Ratio\n(Peak Memory - Input Size) / Input Size × 100%',
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{s:.1f}MB' for s in sizes_valid], rotation=45)
    ax1.set_yscale('log')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    
    # Add value labels
    for bar, val in zip(bars, overhead_pct):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Compression efficiency
    bars2 = ax2.bar(x_pos, compression_valid, color='#9B59B6', alpha=0.8,
                    edgecolor='#2C3E50', linewidth=2)
    
    # Reference line at 1.0 (no compression)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label='No compression (ratio=1.0)')
    
    ax2.set_xlabel('Input Size', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Compression Efficiency\n(Encoded Size / Input Size - Lower is better)',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{s:.1f}MB' for s in sizes_valid], rotation=45)
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    ax2.legend(fontsize=11)
    
    # Add value labels
    for bar, val in zip(bars2, compression_valid):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent / 'memory_efficiency_analysis.png'), dpi=300, bbox_inches='tight')
    print("[OK] Saved: memory_efficiency_analysis.png")
    
    return fig


if __name__ == '__main__':
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*70 + "\n")
    
    plot_memory_scaling()
    plot_memory_distribution()
    plot_scaling_efficiency()
    
    print("\n" + "="*70)
    print("[OK] VISUALIZATIONS COMPLETE")
    print("="*70 + "\n")
    
    print("Generated files:")
    print("  - memory_scaling_analysis.png")
    print("  - memory_distribution_analysis.png")
    print("  - memory_efficiency_analysis.png")
