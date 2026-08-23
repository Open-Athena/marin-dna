## What does this PR do?

<!--
Please include a summary of the change and which issue is fixed.
Please also include relevant motivation and context.
List any dependencies that are required for this change.
List all the breaking changes introduced by this pull request.
-->

Fixes #\<issue_number>

## PR Review Checklist

### Issue Compliance

- [ ] **Issue Compliance Verification**: Review the relevant GitHub issue contents and verify 100% completion of all specified requirements. If any requirement cannot be met, engage with the user immediately to resolve blockers or clarify requirements before proceeding

### Code Quality

- [ ] **Format Code**: Run `make format` to apply consistent formatting (or `pre-commit run -a`)
- [ ] **Simplify Relentlessly**: Verify code follows the simplest design that works - remove unnecessary complexity
- [ ] **DRY Compliance**: Check for code duplication - ensure common functionality is extracted and reused
- [ ] **Code Review**: Review all new/modified code for architectural quality and adherence to project patterns

### Documentation

- [ ] **Design Doc Annotations**: Verify all new/modified files have proper "Design References:" headers (if applicable)
- [ ] **Function Docstrings**: Ensure all non-trivial functions have comprehensive docstrings with type hints
- [ ] **Documentation Updates**: Update relevant documentation (README, AGENTS.md, etc.) if needed

### Testing

- [ ] **Test Implementation Audit**: Scan all test files for:
  - Partially implemented tests
  - Placeholder implementations
  - Mock objects that return fake data
  - `pytest.mark.skip` decorators without justification
  - **All tests must provide real validation with actual implementations**
- [ ] **Integration Tests**: Ensure all tests pass and no warnings are generated (`pytest` or `make test-full`)
- [ ] **Test Coverage**: Verify comprehensive testing of new functionality
- [ ] **Fast Tests**: Run `make test` to ensure fast tests pass during development

### Quality Checks

- [ ] **Pre-commit Hooks**: Run `pre-commit run -a` or `make format` - all hooks must pass
- [ ] **Static Analysis**: Verify code passes all linting and type checking
- [ ] **No Fake Tests**: Confirm no fake implementations, mocks returning fake data, or placeholder tests

### Design Compliance

- [ ] **Architecture**: Verify code follows PyTorch Lightning and Hydra patterns
- [ ] **Config Structure**: Ensure Hydra configs follow project conventions
- [ ] **Integration**: Verify code integrates well with existing systems

## Before Submitting

- [ ] Did you make sure **title is self-explanatory** and **the description concisely explains the PR**?
- [ ] Did you make sure your **PR does only one thing**, instead of bundling different changes together?
- [ ] Did you list all the **breaking changes** introduced by this pull request?
- [ ] Did you **test your PR locally** with `pytest` command?
- [ ] Did you **run pre-commit hooks** with `pre-commit run -a` command?

## Additional Notes

<!-- Any additional context, design decisions, or notes for reviewers -->

## Did you have fun?

Make sure you had fun coding 🙃
