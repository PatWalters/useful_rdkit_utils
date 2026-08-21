# Publishing a release

Releases are built and uploaded by `.github/workflows/publish.yaml` using PyPI
**trusted publishing**, which authenticates GitHub Actions over OpenID Connect.
No API token is stored in the repository, in GitHub secrets, or in a shell
profile.

## One-time setup

This has to be done once per package, by an owner of the PyPI project. It cannot
be done from a workflow.

### 1. PyPI

Go to <https://pypi.org/manage/project/useful-rdkit-utils/settings/publishing/>
and add a trusted publisher with exactly these values:

| Field             | Value                   |
| ----------------- | ----------------------- |
| Owner             | `PatWalters`            |
| Repository name   | `useful_rdkit_utils`    |
| Workflow name     | `publish.yaml`          |
| Environment name  | `pypi`                  |

### 2. TestPyPI (optional but recommended)

Same again at
<https://test.pypi.org/manage/project/useful-rdkit-utils/settings/publishing/>,
with the environment name `testpypi`. This lets you rehearse a release without
consuming a version number on the real index.

### 3. GitHub environments

In **Settings → Environments**, create environments named `pypi` and `testpypi`.

For `pypi`, add yourself under **Required reviewers**. That turns a release into
a job that pauses and waits for you to approve it, which is the last chance to
stop a bad upload — PyPI does not allow a version to be replaced once it is
published.

### 4. Retire the old token

`~/.zshrc` exports `UV_PUBLISH_TOKEN` in plaintext. Once trusted publishing
works, revoke that token at
<https://pypi.org/manage/account/token/> and delete the line, so no long-lived
credential is sitting in a file every process on the machine can read.

## Cutting a release

1. Update `__version__` in `useful_rdkit_utils/__init__.py`. This is the single
   source of truth; hatch reads it at build time.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under the new version.
3. Commit, then tag and push:

   ```shell
   git tag v1.01
   git push origin master --tags
   ```

The workflow then builds the sdist and wheel, checks the tag against the built
version, runs `twine check --strict`, installs the wheel into a clean
environment and imports it, and — after you approve the `pypi` environment —
uploads it.

### Version numbering

Versions are normalised by PEP 440, so `1.00` and `1.0` are the *same* version:
the existing `v1.00` release is `1.0` on PyPI. The tag check compares parsed
versions rather than strings, so either spelling of a tag is accepted. Prefer
plain `1.1`, `1.2` going forward to avoid the ambiguity.

A version cannot be uploaded twice. If a release fails after upload, bump to the
next patch version rather than trying to replace it.

## Rehearsing on TestPyPI

From the **Actions** tab, run **Publish** manually and choose `testpypi`. Then
check the result installs:

```shell
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ useful_rdkit_utils
```

The extra index is needed because TestPyPI does not carry RDKit and the other
runtime dependencies.

## Publishing by hand

If you ever need to bypass the workflow, `uv` reads `UV_PUBLISH_TOKEN`:

```shell
rm -rf dist && python -m build
uv publish
```

Prefer the workflow: it verifies the tag, checks the metadata, and smoke-tests
the wheel before anything reaches the index.
