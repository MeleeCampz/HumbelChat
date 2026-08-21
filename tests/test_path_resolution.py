"""Tests for fix-8: characters.json / KB paths are absolute (CWD-independent)."""
from __future__ import annotations

import importlib
import pathlib

import config.settings as S


class TestAbsolutePathResolution:

    def test_characters_file_is_absolute(self):
        assert S.CHARACTERS_FILE.is_absolute()

    def test_kb_path_is_absolute(self):
        assert S.KB_PATH.is_absolute()

    def test_defaults_point_into_repo(self):
        expected_root = S._REPO_ROOT
        assert S.CHARACTERS_FILE == expected_root / "characters.json"
        assert S.KB_PATH == expected_root / "data" / "knowledge"

    def test_not_dependent_on_cwd(self, monkeypatch, tmp_path):
        """Changing CWD must not change the resolved paths."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("CHARACTERS_FILE", raising=False)
        monkeypatch.delenv("KB_PATH", raising=False)
        importlib.reload(S)
        try:
            assert S.CHARACTERS_FILE.is_absolute()
            assert S.KB_PATH.is_absolute()
            assert S.CHARACTERS_FILE == S._REPO_ROOT / "characters.json"
            assert S.KB_PATH == S._REPO_ROOT / "data" / "knowledge"
        finally:
            importlib.reload(S)

    def test_env_override_respected(self, monkeypatch, tmp_path):
        custom = tmp_path / "my_chars.json"
        monkeypatch.setenv("CHARACTERS_FILE", str(custom))
        importlib.reload(S)
        try:
            assert S.CHARACTERS_FILE == custom
        finally:
            monkeypatch.delenv("CHARACTERS_FILE", raising=False)
            importlib.reload(S)

    def test_empty_env_falls_back_to_default(self, monkeypatch):
        """An empty env var (e.g. a bare `CHARACTERS_FILE=` line) must not
        collapse the path to CWD — it should fall back to the repo default."""
        monkeypatch.setenv("CHARACTERS_FILE", "")
        monkeypatch.setenv("KB_PATH", "")
        importlib.reload(S)
        try:
            assert S.CHARACTERS_FILE == S._REPO_ROOT / "characters.json"
            assert S.KB_PATH == S._REPO_ROOT / "data" / "knowledge"
        finally:
            monkeypatch.delenv("CHARACTERS_FILE", raising=False)
            monkeypatch.delenv("KB_PATH", raising=False)
            importlib.reload(S)
