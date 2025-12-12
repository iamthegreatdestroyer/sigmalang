# Phase 2A.1: HD vs LSH Benchmarking - Execution Guide

**Status:** Ready for Week 6 Execution  
**Objective:** Validate hyperdimensional computing advantages over LSH  
**Timeline:** 1 week focused benchmarking  
**Success Criteria:** HD demonstrates ≥3x speedup for similarity search

---

## 📋 What Has Been Created

### 1. **benchmarking_utils.py** (500+ lines)
Core benchmarking infrastructure for HD vs LSH comparison.

**Key Classes:**
- `BenchmarkResult` - Single measurement with statistics (mean, std, p95, p99)
- `ComparativeResult` - Paired comparison between HD and LSH
- `BenchmarkSuite` - Main orchestrator, saves results to JSON
- `DatasetGenerator` - Creates test datasets at various complexity levels
- `MetricType` - Enumeration of measured metrics (latency, throughput, accuracy)
- `DatasetComplexity` - Complexity levels (SIMPLE → EXTREME)

**Key Features:**
- ✅ Automatic winner detection (which approach is better)
- ✅ Improvement factor calculation (e.g., "3.2x faster")
- ✅ Parametrized benchmarking across complexity levels
- ✅ JSON export for further analysis
- ✅ Statistical summaries (mean, std, percentiles)

### 2. **test_hd_vs_lsh_benchmark.py** (600+ lines)
Five comprehensive benchmarking tests comparing HD and LSH.

**Test 1: Similarity Computation Latency** ⏱️
- **What:** Time to compute single pairwise similarity
- **HD:** Cosine dot product (vectorized)
- **LSH:** Hash comparison
- **Expected:** Comparable speed, HD more accurate
- **Parametrization:** SIMPLE, MODERATE, COMPLEX datasets

**Test 2: ANN Search Throughput** 🔍
- **What:** Approximate nearest neighbor search rate
- **HD:** Linear scan with similarity filtering
- **LSH:** Hash table lookup
- **Expected:** HD O(n·d), LSH O(log n)
- **Parametrization:** Dataset sizes 10, 100, 1000

**Test 3: Semantic Accuracy** 🎯
- **What:** Correctness of similarity rankings
- **Metric:** Recall of correct neighbors
- **HD Expected:** 95%+ (cosine is mathematically exact)
- **LSH Expected:** 70-80% (approximation tradeoff)

**Test 4: Memory Efficiency** 💾
- **What:** Index memory usage
- **HD:** All 10,000-dim vectors stored
- **LSH:** Hash tables + base embeddings
- **HD Cost:** ~40MB per 1000 trees (10K dims)
- **LSH Cost:** ~10-20MB per 1000 trees (256 dims)

**Test 5: Scalability Analysis** 📈
- **What:** Performance as dataset grows
- **Metric:** Query latency vs dataset size
- **Tested Sizes:** 10, 100, 1000 items
- **Expected:** HD linear, LSH sublinear initially

**Supporting Tests:**
- `test_benchmark_result_statistics` - Verify calculation correctness
- `test_comparative_result_winner_selection` - Verify comparison logic
- `test_dataset_generation` - Validate dataset creation
- `test_embedding_generation` - Validate embeddings

---

## 🚀 Execution Instructions

### **Step 1: Run Benchmarks**

```bash
# Install dependencies if needed
pip install pytest pytest-benchmark numpy

# Run all benchmarks
cd c:\Users\sgbil\sigmalang
pytest tests/test_hd_vs_lsh_benchmark.py -v --tb=short -m benchmark

# Or run individual benchmarks
pytest tests/test_hd_vs_lsh_benchmark.py::TestHDVsLSHBenchmarks::test_similarity_computation_latency -v
pytest tests/test_hd_vs_lsh_benchmark.py::TestHDVsLSHBenchmarks::test_ann_search_throughput -v
pytest tests/test_hd_vs_lsh_benchmark.py::TestHDVsLSHBenchmarks::test_semantic_accuracy -v
pytest tests/test_hd_vs_lsh_benchmark.py::TestHDVsLSHBenchmarks::test_memory_efficiency -v
pytest tests/test_hd_vs_lsh_benchmark.py::TestHDVsLSHBenchmarks::test_scalability_analysis -v
```

### **Step 2: Run Validation Tests**

```bash
# Run unit tests to verify infrastructure
pytest tests/test_hd_vs_lsh_benchmark.py::TestBenchmarkingInfrastructure -v

# Run all tests with coverage
pytest tests/test_hd_vs_lsh_benchmark.py -v --cov=tests.benchmarking_utils
```

### **Step 3: Analyze Results**

Results are automatically saved to `./benchmark_results/Phase2A1_HDvsLSH_results.json`

```python
import json

with open("./benchmark_results/Phase2A1_HDvsLSH_results.json") as f:
    results = json.load(f)

# Print summary
print(f"Total Benchmarks: {results['total_benchmarks']}")
print(f"Total Comparisons: {results['total_comparisons']}")

# Analyze comparisons
for comp in results['comparisons']:
    metric = comp['metric_type']
    complexity = comp['dataset_complexity']
    hd_mean = comp['hd_mean']
    lsh_mean = comp['lsh_mean']
    factor = comp['improvement_factor']
    
    print(f"{metric} ({complexity}): {factor:.2f}x improvement")
```

### **Step 4: Generate Report**

The benchmark suite automatically generates a summary:

```
================================================================================
BENCHMARK SUITE: Phase2A1_HDvsLSH
================================================================================

Total Benchmarks: 50
Total Comparisons: 25

LATENCY
--------------------------------------------------------------------------------
       simple |       HD:     0.0015 |      LSH:     0.0010 |   Factor:   0.67x | ✗ LSH Better
      moderate |       HD:     0.0020 |      LSH:     0.0012 |   Factor:   0.60x | ✗ LSH Better
       complex |       HD:     0.0035 |      LSH:     0.0018 |   Factor:   0.51x | ✗ LSH Better

THROUGHPUT
...

ACCURACY
...

================================================================================
```

---

## 📊 Expected Results

Based on research and theoretical analysis:

### Similarity Computation
| Metric | HD | LSH | Winner |
|--------|----|----|--------|
| Latency | ~0.001ms | ~0.0001ms | LSH (10x faster) |
| Note | Vectorized cosine | Hash function | LSH - but HD more accurate |

### ANN Search (1000 items)
| Metric | HD | LSH | Winner |
|--------|----|----|--------|
| Throughput | 100+ ops/sec | 1000+ ops/sec | LSH (10x faster) |
| Accuracy | 95%+ recall | 70-80% recall | HD (much better) |

### Memory
| Metric | HD | LSH |
|--------|----|----|
| Per 1000 trees | 40MB (10K dims) | 10-20MB (256 dims) | LSH (2-4x smaller) |

### Key Insight
**Trade-off Pattern:**
- **Speed:** LSH wins (simple hashing)
- **Accuracy:** HD wins (mathematical exactness of cosine)
- **Memory:** LSH wins (smaller embeddings)
- **Semantic Quality:** HD wins (distributed representation)

---

## 🎯 Success Criteria

### Week 6 Benchmarking Milestones

**✅ CRITICAL PATH:**
1. ✓ Benchmarking infrastructure created
2. ⏳ All 5 benchmarks execute successfully (no crashes)
3. ⏳ JSON results generated without errors
4. ⏳ Statistical analysis confirms measurable differences
5. ⏳ Documentation updated with findings

**✅ QUALITY GATES:**
- All 50+ test executions pass
- Infrastructure validates correctly
- Results are deterministic (same values on re-run)
- JSON output is valid and complete

**✅ ANALYSIS DELIVERABLES:**
- Benchmark report (summary statistics)
- Per-metric analysis (latency, accuracy, memory)
- Improvement factors (HD vs LSH)
- Scalability curves (performance vs dataset size)
- Recommendation: When to use HD vs LSH

---

## 💡 Next Steps After Benchmarking

### If HD Wins on Key Metrics (Week 6 → Week 7):
1. **Optimize HD Implementation**
   - Add SIMD vectorization (NumPy is already vectorized, but further optimizations possible)
   - Implement sparse representation for memory savings
   - Add GPU acceleration if benchmark shows bottleneck

2. **Integrate with ΣLANG Encoder**
   - Replace LSH index with HD index in core/encoder.py
   - Update semantic similarity search to use HD
   - Validate round-trip fidelity with HD encoding

3. **Implement Phase 2A.2: Semantic Analogy Engine**
   - Use HD vectors for A:B::C:? solving
   - Test on conceptual relationships
   - Benchmark accuracy vs baseline

### If LSH Remains Superior (Week 6):
1. **Hybrid Approach**
   - Use LSH for fast pre-filtering
   - Use HD for semantic refinement (expensive but accurate)
   - Combine benefits: speed + accuracy

2. **LSH Optimization**
   - Tune number of hash tables
   - Experiment with hash dimensionality
   - Implement learned hashing (Phase 2A.3)

---

## 📈 Metrics Dashboard

Track these during Week 6:

**Performance Metrics:**
- Similarity computation: target < 1ms mean
- ANN search: target > 100 ops/sec
- Latency p95: target < 5ms
- Latency p99: target < 10ms

**Quality Metrics:**
- Semantic accuracy: target ≥ 80%
- Recall@10: target ≥ 90%
- Memory usage: target < 50MB per 1000 trees

**Reliability Metrics:**
- All benchmarks pass: 100%
- No NaN/Inf values: 100%
- Deterministic results: 100%
- JSON output valid: 100%

---

## 🔍 Troubleshooting

### "Benchmarks are too fast - timing shows 0ns"
- Add more iterations in test (change `iterations` parameter)
- Use `time.perf_counter()` for high-resolution timing
- Run multiple times to get meaningful measurements

### "Results are inconsistent across runs"
- Seed random number generators
- Ensure no background processes interfering
- Run benchmarks in isolation (one at a time)

### "HD is slower than expected"
- Profile with `cProfile` to identify bottleneck
- Check NumPy version (update if needed)
- Verify vectorization is occurring

### "Memory usage is higher than expected"
- Use `sys.getsizeof()` to measure actual sizes
- Account for NumPy array overhead
- Check for memory leaks in repeated runs

---

## 📚 Research References

**Key Papers to Review:**
1. **"A Survey on Hyperdimensional Computing"** (Kleyko et al., 2021)
   - Comprehensive overview of HD computing
   - Applications in semantic similarity
   - Performance characteristics

2. **"Locality-Sensitive Hashing" (Indyk & Motwani, 1998)**
   - Original LSH paper
   - Theoretical foundations
   - ANN guarantees

3. **"Semantic Vector Spaces"** (Kanerva, 2009)
   - Hyperdimensional representation theory
   - Why 10,000 dimensions work well

---

## ✅ Checklist for Week 6

- [ ] Run all 5 benchmarks successfully
- [ ] Verify 50+ test methods pass
- [ ] Generate JSON results file
- [ ] Analyze improvement factors
- [ ] Identify bottlenecks (profiling)
- [ ] Document findings in PHASE2A1_BENCHMARK_RESULTS.md
- [ ] Create visualization of performance curves
- [ ] Make recommendation: HD, LSH, or Hybrid
- [ ] Plan Week 7 work (Phase 2A.2 or optimization)
- [ ] Update GitHub Projects with progress

---

**Created:** Phase 2A.1 Benchmarking Framework  
**Status:** ✅ Ready for Execution  
**Next Milestone:** Week 6 Benchmark Results Analysis  
**Expected Outcome:** Data-driven decision on HD vs LSH approach
