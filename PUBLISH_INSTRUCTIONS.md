# PyPI Publish Instructions

Wheel is built and checked. To publish:

```bash
pip install build twine

# Build (already done — produces dist/sigmalang-2.0.0-py3-none-any.whl)
python -m build

# Verify
python -m twine check dist/*

# Upload to PyPI (requires API token)
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-<your-token-here> python -m twine upload dist/*

# Or for TestPyPI first
python -m twine upload --repository testpypi dist/*
```

The package name on PyPI will be `sigmalang`. Ensure the PyPI project `sigmalang` is claimed under the `iamthegreatdestroyer` account before upload.
