---
name: Bug Report
about: Report a bug or unexpected behavior
title: "[BUG] "
labels: bug
---

## Bug Description

<!-- Provide a clear and concise description of the bug -->

## Steps to Reproduce

<!-- Detailed steps to reproduce the behavior -->

1. Run command: `...`
2. With config: `...`
3. See error: `...`

## Expected Behavior

<!-- What should happen instead? -->

## Actual Behavior

<!-- What actually happens? Include error messages, stack traces, etc. -->

## Environment

<!-- Provide relevant environment information -->

- Python version: `python --version`
- OS: [e.g., Linux, macOS, Windows]
- PyTorch version: `python -c "import torch; print(torch.__version__)"`
- Lightning version: `python -c "import lightning; print(lightning.__version__)"`
- Hydra version: `python -c "import hydra; print(hydra.__version__)"`

## Configuration

<!-- If applicable, include the relevant config file or command-line overrides -->

```yaml
# configs/example.yaml
key: value
```

Or command:
```bash
python glm_experiments/train.py trainer.max_epochs=10 data.batch_size=32
```

## Error Messages / Stack Traces

<!-- Paste full error messages or stack traces here -->

```
Traceback (most recent call last):
  File "...", line X, in ...
    ...
Error: ...
```

## Root Cause Analysis

<!-- If you've investigated, describe what you think is causing the bug -->

## Proposed Fix

<!-- Describe how you plan to fix the bug -->

### Implementation Plan

1. [ ] Step 1: Description
2. [ ] Step 2: Description

### Test Requirements

<!-- How will you verify the fix works? Remember: real tests only! -->

- [ ] Reproduce the bug in a test
- [ ] Verify the fix resolves the issue
- [ ] Ensure no regressions are introduced
- [ ] Test edge cases related to the bug

## Files to Modify

<!-- List files that need to be changed -->

- `path/to/file.py` - Description of changes

## Additional Context

<!-- Add any other context, screenshots, or examples about the bug here -->

## Checklist

Before submitting, ensure:
- [ ] Bug description is clear
- [ ] Steps to reproduce are detailed
- [ ] Expected vs actual behavior is documented
- [ ] Environment information is provided
- [ ] Proposed fix includes test requirements
