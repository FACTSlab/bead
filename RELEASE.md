# Release Process for bead

This document describes the manual steps required to set up releases for the `bead` package under the `caulking` GitHub organization.

## One-Time Setup

### 1. Create the GitHub Organization

1. Go to https://github.com/organizations/new
2. Create organization named `caulking`
3. Add yourself as an owner

### 2. Transfer Repository

Option A: Transfer existing repository:
1. Go to https://github.com/aaronstevenwhite/bead/settings
2. Scroll to "Danger Zone"
3. Click "Transfer repository"
4. Enter `caulking` as the new owner
5. Confirm the transfer

Option B: Create new repository and push:
```bash
# Create new repo at github.com/caulking/bead
# Then update remote:
git remote set-url origin https://github.com/caulking/bead.git
git push -u origin main
```

### 3. Enable GitHub Pages

1. Go to https://github.com/caulking/bead/settings/pages
2. Under "Build and deployment":
   - Source: "GitHub Actions"
3. The docs workflow will automatically deploy on push to main

### 4. Set Up PyPI Trusted Publishing

PyPI now supports "Trusted Publishing" which allows GitHub Actions to publish without API tokens.

#### 4.1 Create PyPI Account (if needed)

1. Go to https://pypi.org/account/register/
2. Create account and verify email
3. Enable 2FA (required for publishing)

#### 4.2 Register the Package Name

Before the first release, you must register the package name on PyPI:

1. Go to https://pypi.org/manage/account/publishing/
2. Click "Add a new pending publisher"
3. Fill in:
   - PyPI Project Name: `bead`
   - Owner: `caulking`
   - Repository name: `bead`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
4. Click "Add"

#### 4.3 Set Up TestPyPI (Optional but Recommended)

1. Go to https://test.pypi.org/manage/account/publishing/
2. Add the same pending publisher with environment name: `testpypi`

### 5. Configure GitHub Environments

1. Go to https://github.com/caulking/bead/settings/environments
2. Create environment named `pypi`:
   - Add deployment protection rules if desired (e.g., required reviewers)
3. Create environment named `testpypi`:
   - No protection rules needed (for testing)

### 6. Verify CI Workflow

After transferring, verify CI passes:
```bash
git push origin main
```

Check https://github.com/caulking/bead/actions to ensure all workflows run successfully.

---

## Creating a Release

### 1. Update Version Number

Update version in both files (they must match):

**pyproject.toml:**
```toml
[project]
version = "X.Y.Z"
```

**bead/__init__.py:**
```python
__version__ = "X.Y.Z"
```

### 2. Update Changelog (Optional)

If you maintain a CHANGELOG.md, update it with the new version's changes.

### 3. Commit Version Bump

```bash
git add pyproject.toml bead/__init__.py
git commit -m "Bump version to X.Y.Z"
git push origin main
```

### 4. Create Git Tag

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

### 5. Monitor Release

1. Go to https://github.com/caulking/bead/actions
2. Watch the "Publish to PyPI" workflow
3. It will:
   - Build the distribution
   - Publish to TestPyPI (for verification)
   - Publish to PyPI

### 6. Verify Release

After the workflow completes:

```bash
# Verify on PyPI
pip index versions bead

# Test installation
pip install bead==X.Y.Z
python -c "import bead; print(bead.__version__)"
```

### 7. Create GitHub Release (Optional)

1. Go to https://github.com/caulking/bead/releases/new
2. Select the tag `vX.Y.Z`
3. Generate release notes or write custom notes
4. Publish release

---

## Version Numbering

Follow semantic versioning (SemVer):

- **MAJOR** (X.0.0): Breaking API changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

Current version: **0.1.0** (initial release)

---

## Troubleshooting

### PyPI Publishing Fails with "Invalid token"

This usually means trusted publishing is not configured correctly:
1. Verify the pending publisher on PyPI matches exactly:
   - Owner: `caulking`
   - Repository: `bead`
   - Workflow: `publish.yml`
   - Environment: `pypi`

### Docs Not Deploying

1. Check GitHub Pages is enabled in repository settings
2. Verify the `docs.yml` workflow has `pages: write` permission
3. Check the Actions tab for workflow errors

### Version Mismatch

If `pip install bead` shows wrong version:
1. Verify both `pyproject.toml` and `bead/__init__.py` have the same version
2. Clear pip cache: `pip cache purge`
3. Reinstall: `pip install --no-cache-dir bead==X.Y.Z`

---

## First Release Checklist

- [ ] Create `caulking` organization on GitHub
- [ ] Transfer or create `bead` repository under `caulking`
- [ ] Enable GitHub Pages (Source: GitHub Actions)
- [ ] Create PyPI account with 2FA enabled
- [ ] Add pending publisher on PyPI for `bead`
- [ ] Create `pypi` environment in GitHub repo settings
- [ ] Verify CI workflow passes
- [ ] Create tag `v0.1.0`
- [ ] Verify docs deploy to https://caulking.github.io/bead
- [ ] Verify package available on PyPI
- [ ] Test `pip install bead`
