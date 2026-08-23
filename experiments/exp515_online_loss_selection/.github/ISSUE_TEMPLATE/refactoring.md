---
name: Refactoring
about: Propose code refactoring or restructuring
title: "[REFACTOR] "
labels: refactoring
---

## Refactoring Goal

<!-- What is the goal of this refactoring? What problem does it solve? -->

## Current State

<!-- Describe the current code structure and what needs improvement -->

### Current Design
```python
# Example of current code structure
```

### Issues with Current Design
- Issue 1: Description
- Issue 2: Description

## Proposed State

<!-- Describe the desired code structure after refactoring -->

### New Design
```python
# Example of proposed code structure
```

### Benefits
- Benefit 1: Description
- Benefit 2: Description

## Implementation Plan

<!-- Break down refactoring into small, safe steps. Each step should maintain working code. -->

### Step 1: [Description]
<!-- What will be done in this step -->
- [ ] Task 1
- [ ] Task 2
- [ ] Verify: All tests still pass

### Step 2: [Description]
- [ ] Task 1
- [ ] Task 2
- [ ] Verify: All tests still pass

## Test Strategy

<!-- How will you ensure refactoring doesn't break functionality? -->

- [ ] All existing tests pass before refactoring
- [ ] Add tests for any new behavior introduced
- [ ] Integration tests verify end-to-end functionality
- [ ] No test mocks/fakes - use real implementations

## Files Affected

### Files to Modify
- `path/to/file1.py` - Description of changes
- `path/to/file2.py` - Description of changes

### Files to Create
- `path/to/new_file.py` - Description

### Files to Delete
- `path/to/old_file.py` - Reason for deletion

## Breaking Changes

<!-- List any breaking changes and how they will be handled -->

- [ ] Change 1: Description and migration path
- [ ] Change 2: Description and migration path

## Design Principles Applied

<!-- Which design principles from CLAUDE.md are being applied? -->

- [ ] Simplify Relentlessly
- [ ] DRY (Don't Repeat Yourself)
- [ ] Other: Description

## Risk Assessment

<!-- What are the risks of this refactoring? How will they be mitigated? -->

- Risk 1: Mitigation strategy
- Risk 2: Mitigation strategy

## Additional Context

<!-- Add any other context, diagrams, or examples about the refactoring -->

## Checklist

Before submitting, ensure:
- [ ] Refactoring goal is clear
- [ ] Current and proposed states are documented
- [ ] Implementation plan breaks down into safe steps
- [ ] Test strategy ensures no regressions
- [ ] Breaking changes are identified and documented
- [ ] Risk assessment is complete
