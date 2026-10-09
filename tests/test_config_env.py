"""AxiomConfig.from_env: presets and per-field overrides."""

import pytest

from axiom_engine.config import AxiomConfig


def test_default_unchanged_without_env(monkeypatch):
    for k in ("AXIOM_PY_PRESET", "AXIOM_PY_D_MODEL", "AXIOM_PY_LR_INNER"):
        monkeypatch.delenv(k, raising=False)
    assert AxiomConfig.from_env() == AxiomConfig()


def test_tiny_preset_is_small(monkeypatch):
    monkeypatch.setenv("AXIOM_PY_PRESET", "tiny")
    cfg = AxiomConfig.from_env()
    assert cfg.d_model == 64 and cfg.n_layers == 2
    assert cfg.d_model % cfg.num_heads == 0


def test_field_override_wins(monkeypatch):
    monkeypatch.setenv("AXIOM_PY_PRESET", "tiny")
    monkeypatch.setenv("AXIOM_PY_D_MODEL", "128")
    monkeypatch.setenv("AXIOM_PY_LR_INNER", "0.01")
    cfg = AxiomConfig.from_env()
    assert cfg.d_model == 128
    assert cfg.lr_inner == pytest.approx(0.01)


def test_unknown_preset_rejected(monkeypatch):
    monkeypatch.setenv("AXIOM_PY_PRESET", "huge")
    with pytest.raises(ValueError):
        AxiomConfig.from_env()
