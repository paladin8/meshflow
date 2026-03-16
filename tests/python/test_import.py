"""Smoke test: verify meshflow package imports and Rust extension is available."""


def test_import_meshflow() -> None:
    import meshflow

    assert meshflow.__version__ == "0.1.0"


def test_rust_runtime_version() -> None:
    from meshflow import runtime_version

    assert runtime_version() == "0.1.0"
