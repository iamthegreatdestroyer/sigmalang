# 1M Context Window Analysis Prompts

After loading full codebase, use these prompts:

## Dependency Graph Analysis
```
Using full ΣLANG codebase:
1. Generate dependency graph (core → sigmalang → tests)
2. Identify circular dependencies
3. Create Mermaid diagram
4. Highlight Phase 2 test gaps
```

## Test Coverage Analysis
```
Analyze test coverage:
1. List all modules in core/ and sigmalang/
2. Identify existing tests
3. Calculate coverage %
4. Generate missing test stubs
```

## Performance Bottleneck Detection
```
Identify bottlenecks:
1. O(n²) or higher algorithms
2. Inefficient data structures
3. Unnecessary allocations
4. Cross-reference with benchmarks
```

## Security Audit (@CIPHER)
```
@CIPHER audit full codebase:
1. Review crypto_primitives.py, key_derivation.py
2. Check deprecated algorithms
3. Verify constant-time operations
4. Generate security report
```
