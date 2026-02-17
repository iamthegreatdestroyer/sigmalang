# Claude Code Execution Integration Tests

Tests designed for Claude's sandboxed Python environment.

## Quick Start (In Claude Chat)

### Quick Validation (5 seconds)
```
Run S:\sigmalang\tests\claude_integration\quick_validate.py
```

### Full Suite (30 seconds)
```
Run S:\sigmalang\tests\claude_integration\test_code_execution.py
```

## What These Validate
1. ΣLANG modules accessible in Claude sandbox
2. Encode pipeline works correctly
3. Decode pipeline accurate
4. Semantic fidelity >0.85 maintained
5. Compression ratio 70% achieved
6. Critical tokens never modified

## Expected Output
```
[PASS] basic_encode_decode: PASS
   similarity: 0.891
   compression_ratio: 0.284

[PASS] preserve_critical_tokens: PASS
```

## Troubleshooting
- **Import Error:** Copy core modules to /tmp/ in sandbox
- **Missing Dependencies:** `pip install torchhd numpy scipy` in sandbox
- **Path Issues:** Verify `sys.path.insert(0, 'S:/sigmalang')` is first line
