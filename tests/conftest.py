"""Shared pytest configuration."""
from __future__ import annotations


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (seconds to minutes); run explicitly with "
        "`pytest -m slow` or via -k selector.",
    )
