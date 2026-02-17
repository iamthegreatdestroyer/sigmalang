# ΣLANG Model Selection Strategy

## Model Characteristics

| Model | Tokens | Speed | Cost | Best For |
|-------|--------|-------|------|----------|
| Haiku 4.5 | ~8K | 2x | 1/3 | Validation, tests |
| Sonnet 4.5 | ~15K | 1x | 1x | Code generation |
| Opus 4.5 | ~7.5K | 0.8x | 3x | Optimization |
| Opus 4.6 | ~7.5K | 0.8x | 3x | Agent teams |

## Task Routing

**Use Haiku 4.5:**
- Integration test validation
- Code linting
- Quick syntax checks

**Use Sonnet 4.5:**
- Feature implementation
- Test generation
- Bug fixes

**Use Opus 4.5:**
- Architecture decisions
- Performance optimization
- Security analysis

**Use Opus 4.6:**
- Multi-agent orchestration
- Phase planning

## VS Code Usage

Specify model explicitly:
```
Using Claude Haiku 4.5:
Run integration tests
```

```
Using Claude Opus 4.5:
Optimize encoder.py for 2x throughput
```

## Cost Optimization

Smart routing saves 70%:
- All Opus: 100 tasks × $90 = $9,000
- Smart routing: $1,728 (81% savings)
