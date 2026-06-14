# Commit Standard For SMap

This document defines the official commit standard for SMap. The goal is to keep
the project history readable, reviewable, and auditable.

## Format

SMap uses Conventional Commits:

```text
type(scope): subject
```

Keep the subject at or below 72 characters when possible. Wrap body lines at
roughly 72-88 characters.

Examples from the normalized history:

```text
chore(project): initialize repository and early structure
feat(core): implement initial debugging UI and core logic stubs
build(ci): establish build system and CI foundation
ci(test): integrate initial tests and refine workflows
test(ci): add visual tests and enhance CI publishing
refactor(release): prepare for v1.0.0-rc.1 and cleanup assets
feat(core): refine SMap algorithm and calibration notebook
fix(docs): apply v1.0.4 hotfix and finalize documentation
build(deps): prepare for the new update and cleanup codebase
```

## Types

- `feat`: add a new feature or capability.
- `fix`: fix a bug or behavioral mismatch.
- `docs`: change documentation, README files, tutorial notebooks, or
  documentation images only.
- `test`: add or update tests.
- `ci`: change GitHub Actions or pipeline behavior.
- `build`: change packaging, dependencies, release setup, or build tooling.
- `refactor`: restructure code without changing behavior.
- `chore`: maintain repository internals that do not fit the other types.
- `perf`: improve performance without changing the public API.
- `style`: change formatting without changing logic.

## Recommended Scopes

- `core`: `smap/`, algorithms, and public API.
- `docs`: `README.md`, `docs/`, tutorial documentation.
- `test`: `tests/`, test data, visual tests.
- `ci`: `.github/workflows/`.
- `deps`: `requirements.txt`, `environment.yml`, dependency setup.
- `release`: versions, tags, PyPI, release artifacts.
- `hdvo`: `tools/testing/hdvo/`.
- `vtest`: `tools/testing/vtest/`.
- `assets`: logo, images, and media.
- `project`: repository initialization or broad project structure.

## Subject Rules

- Use a concise imperative verb such as `add`, `fix`, `refactor`, or `update`
  when it fits the change.
- Write subjects in English to match the current history.
- Do not end the subject with a period.
- Avoid generic messages such as `Update README.md`, `Add files via upload`, or
  `Maintain unittest-passed status`.
- If a commit touches multiple files, describe the purpose rather than the file
  operation.

## Body Rules

A body is optional, but recommended when a commit:

- Changes algorithm behavior.
- Affects releases, versions, or licenses.
- Groups several related changes under one purpose.
- Records a migration, conflict resolution, or tradeoff that should remain
  auditable later.

Body template:

```text
Explain why the change is needed and what behavior it preserves.

Refs: #issue-number
```

For larger commits, the body should answer three questions:

- What problem or goal does this commit address?
- What state or behavior does it preserve?
- What tradeoff or resolution should future reviewers know about?

## Verification

Before pushing, run at least the unit tests:

```bash
python -m unittest discover -s tests -p "[vu]test*.py"
```

## CI/CD

The repository already has unit-test CI. Lint is introduced in two phases to
avoid making CI fail abruptly on the existing baseline:

1. Soft gate: `lint.yml` runs `pylint` and allows failure while the baseline is
   being measured.
2. Hard gate: after the lint baseline is fixed, remove `continue-on-error` and
   require lint to pass for pull requests.

The commit-standards workflow validates commit subjects on PRs and pushes using
the Conventional Commits format. If remote GPG-key enforcement is needed, prefer
GitHub branch protection and verified-signature policies instead of trying to
verify signatures inside a runner without a project keyring.

After the lint baseline is healthy, promote lint from soft gate to hard gate.
