"""
WORKSTREAM C: Memory Profiling Quick Start
============================================

@PRISM: Data Science & Statistical Analysis

Quick commands to run memory profiling and analysis.
"""

# ============================================================================
# QUICK START GUIDE
# ============================================================================

# 1. Install required packages (one time)
# pip install memory-profiler psutil scipy pandas seaborn matplotlib

# 2. Run baseline profiling (Phase 4A.2 current state)
# cd tests/
# python run_memory_baseline.py

# 3. View results
# cat memory_profiles/baseline_summary.json

# 4. Run statistical analysis
# python memory_analysis.py

# 5. Compare visualizations
# - memory_analysis_results/memory_scaling.png     : Memory vs file size
# - memory_analysis_results/distributions.png      : Statistical distributions
# - memory_analysis_results/scaling_ratios.png     : Scaling efficiency

# ============================================================================
# EXPECTED OUTCOMES
# ============================================================================

"""
Success Criteria (from brief):
✅ Peak memory < 500MB for 100MB input
✅ Linear or sub-linear scaling with file size
✅ Memory freed promptly after encoding
✅ Identify < 1% memory leaks (acceptable)
✅ 95% confidence in measurements

Baseline Phase 4A.2 expectations:
- Small files (1KB): ~2-5 MB peak
- Medium files (1MB): ~20-50 MB peak
- Large files (10MB): ~100-200 MB peak
- Extra large (100MB): ~300-400 MB peak

Scaling behavior:
- Ideal: Linear (memory ≈ input size, ratio ~1.0x)
- Good: Sub-linear (memory < input size, ratio <1.0x)
- Acceptable: 2-3x linear (ratio <3.0x)
- Concerning: >3x linear (indicates algorithm issue)
"""

# ============================================================================
# DATA INTERPRETATION GUIDE
# ============================================================================

"""
STATISTICAL METRICS EXPLAINED:

Peak Memory:
  - Maximum RSS memory during encoding
  - Most important metric for success criteria
  - Should be stable across runs (low std dev)

Scaling Ratio:
  - peak_memory_mb / file_size_mb
  - 0.5x = memory < input (excellent)
  - 1.0x = memory ≈ input (good)
  - 2.0x = memory ≈ 2x input (acceptable)
  - >3.0x = problematic, investigate algorithm

95% Confidence Interval (CI):
  - Range of peak memory at 95% confidence
  - Narrow CI = consistent behavior
  - Wide CI = high variance, needs investigation

Compression Ratio:
  - encoded_size / input_size
  - SigmaLang typically: 0.1-0.3x (10-30% of original)
  - Validates encoding is actually compressing

Memory Freed:
  - initial_memory - final_memory
  - Should be close to peak_memory (memory returned to OS)
  - <50% freed suggests memory retention issue

Anomalies:
  - Runs that exceed 1.5x IQR bounds
  - May indicate GC timing, OS interference
  - Investigate if systematic pattern

Normality Test (Shapiro-Wilk):
  - p > 0.05 = normal distribution (good for CI)
  - p < 0.05 = non-normal (may need non-parametric CI)

Trend Test:
  - Slope < 0 = memory decreasing across runs (good)
  - Slope > 0 = memory increasing (memory leak?)
  - p < 0.05 = significant trend
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
High peak memory (>500MB for 100MB input):
  1. Check if GC is disabled
  2. Verify no intermediate copies being made
  3. Profile with py-spy to find hotspots
  4. Check for memory leaks with tracemalloc

Inconsistent measurements (high std dev):
  1. Run more iterations (increase NUM_RUNS)
  2. Close background applications
  3. Disable CPU scaling
  4. Use nice -n -20 to increase priority

Non-linear scaling:
  1. Check algorithm complexity (should be O(n))
  2. Look for redundant copying
  3. Verify streaming works for large files
  4. Check if buffer pool is functioning

Memory not freed:
  1. Check if cache isn't cleared
  2. Verify encoding state is garbage collected
  3. Look for circular references
  4. Force gc.collect() between runs
"""

# ============================================================================
# FILES GENERATED
# ============================================================================

"""
After running baseline profiling:

memory_profiles/
  ├── baseline_summary.json           # Summary statistics for all sizes
  ├── profile_size_0.01MB_run_0.json  # Individual profile (10B)
  ├── profile_size_1.00MB_run_0.json  # Individual profile (1MB)
  └── ...

memory_analysis_results/
  ├── memory_scaling.png              # Main plot: memory vs input size
  ├── distributions.png               # Statistical distributions
  ├── scaling_ratios.png              # Efficiency analysis
  └── memory_analysis_TIMESTAMP.json  # Full analysis with tests
"""

# ============================================================================
# SAMPLE ANALYSIS OUTPUT
# ============================================================================

"""
Expected structure of baseline_summary.json:

{
  "timestamp": "2025-12-13T15:30:00",
  "platform": {
    "os": "nt",
    "system": "win32",
    "memory_available_gb": 15.2
  },
  "results": {
    "0.01": {
      "file_size_mb": 0.01,
      "num_runs": 5,
      "peak_memory": {
        "mean_mb": 3.5,
        "std_mb": 0.2,
        "min_mb": 3.2,
        "max_mb": 3.8,
        "ci_95_lower": 3.2,
        "ci_95_upper": 3.8
      },
      "scaling_ratio": {
        "mean": 350.0,
        "std": 20.0
      },
      "compression_ratio": 0.15
    },
    "1.0": {
      "file_size_mb": 1.0,
      "num_runs": 5,
      "peak_memory": {
        "mean_mb": 25.3,
        "std_mb": 0.8,
        "min_mb": 24.2,
        "max_mb": 26.5,
        "ci_95_lower": 24.0,
        "ci_95_upper": 26.6
      },
      "scaling_ratio": {
        "mean": 25.3,
        "std": 0.8
      },
      "compression_ratio": 0.14
    }
  }
}
"""

print(__doc__)
