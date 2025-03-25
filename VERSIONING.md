# Versioning Guidelines for SMap

Our project SMap follows [Semantic Versioning (SemVer)](https://semver.org/) as well as Python’s PEP 440 guidelines. This document outlines the version format and the rules for incrementing version numbers with each release.

## 1. Version Format

All versions must follow the format:

```
MAJOR.MINOR.PATCH
```

- **MAJOR:** Incremented when there are incompatible API changes (breaking changes).
- **MINOR:** Incremented when functionality is added in a backward-compatible manner.
- **PATCH:** Incremented for backward-compatible bug fixes and minor improvements.

## 2. When to Bump Each Version Component

### PATCH Version
- **When to bump:** For bug fixes, performance improvements, or minor changes that do not affect the public API.
- **Example:** `0.1.0` → `0.1.1`

### MINOR Version
- **When to bump:** When adding new features or modules that remain backward-compatible.
- **Example:** `0.1.1` → `0.2.0`

### MAJOR Version
- **When to bump:** When introducing significant changes that break backward compatibility.
- **Example:** `0.2.0` → `1.0.0`

### Pre-release Versions (Optional)
- For testing purposes, you may include pre-release identifiers like `alpha`, `beta`, or `rc` (release candidate).
- **Example:** `1.0.0rc1`, `1.0.0-beta.2`

## 3. Release Process

1. **Update the Version:**  
   Update the version in `smap/__init__.py` (or use an automated tool like [setuptools_scm](https://pypi.org/project/setuptools-scm/) to derive the version from Git tags).

2. **Create a Git Tag:**  
   Use the format `vMAJOR.MINOR.PATCH` (e.g., `v0.1.1`) to create a tag in Git.

3. **Create a GitHub Release:**  
   Based on the tag, create a GitHub Release that includes additional details (changelog, release notes, etc.).

4. **CI/CD Build:**  
   Every release build should run unit tests via our CI/CD pipeline (e.g., GitHub Actions). If tests fail, the release is rejected and the tag is not created.

## 4. Automating Versioning (Optional)

If you wish to have an automated process that increments the PATCH version on every merge or daily build, consider using a tool like [setuptools_scm](https://pypi.org/project/setuptools-scm/) to generate versions based on Git commit counts or dates. **However**, ensure that:
- The PATCH version is only incremented for actual bug fixes (if merges are trivial refactorings, consider bumping the MINOR version instead).
- The automated process allows manual override so you can set the version manually when necessary.

## 5. Examples

- **Bug Fix:** Update from `0.1.0` to `0.1.1`
- **New Feature (backward-compatible):** Update from `0.1.1` to `0.2.0`
- **Breaking Change:** Update from `0.2.0` to `1.0.0`
