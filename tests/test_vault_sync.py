"""Tests for bot_core.vault_sync (#19) — Obsidian vault ↔ KB git sync.

Git is faked by patching ``vault_sync._run_git`` (the single subprocess
boundary), so no real git binary or network is needed.  Covers: config
parsing, clone-once behavior, one-way mirror pulls, two-way campaign
commit/push (incl. push-reject retry and conflict abort), and the startup
loop's re-index trigger + failure isolation.
"""
from __future__ import annotations

import pathlib
from unittest.mock import AsyncMock, patch

import pytest

from bot_core import vault_sync


# ──────────────────────────── Fake git ────────────────────────────

class FakeGit:
    """Dispatches the git argv used by vault_sync; records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []
        self.heads: dict[str, str] = {}      # cwd(str) → HEAD sha
        self.remote_head = "remote-1"
        self.dirty: set[str] = set()         # cwds with uncommitted changes
        self.fail_fetch = False
        self.conflict_pull = False           # one-shot pull rebase conflict
        self.push_reject_once = False        # one-shot non-fast-forward push

    def __call__(self, args: list[str], cwd: pathlib.Path) -> tuple[int, str]:
        key = str(cwd)
        self.calls.append((list(args), key))
        cmd = args[0]
        if cmd == "clone":
            self.heads[key] = self.remote_head
            return (0, "")
        if cmd == "config":
            return (0, "")
        if cmd == "rev-parse":
            if args[1] == "--abbrev-ref":
                # Any dir with a .git counts as a checkout (make_checkout).
                return (0, "main") if (pathlib.Path(cwd) / ".git").exists() else (1, "unknown revision")
            return (0, self.heads.get(key, "h0"))  # HEAD
        if cmd == "fetch":
            if self.fail_fetch:
                return (1, "fatal: auth failed")
            return (0, "")
        if cmd in ("reset", "pull"):
            if cmd == "pull" and self.conflict_pull:
                self.conflict_pull = False
                return (1, "CONFLICT (content): Merge conflict in notes.md")
            self.heads[key] = self.remote_head
            return (0, "")
        if cmd == "rebase":  # --abort
            return (0, "")
        if cmd == "status":
            return (0, " M session_notes/x/notes.md\n" if key in self.dirty else "")
        if cmd == "add":
            return (0, "")
        if cmd == "commit":
            self.dirty.discard(key)
            self.heads[key] = f"local-{len(self.calls)}"
            return (0, "")
        if cmd == "push":
            if self.push_reject_once:
                self.push_reject_once = False
                return (1, "! [rejected] main -> main (non-fast-forward)")
            return (0, "")
        return (1, f"unexpected git command: {cmd}")

    def ran(self, *prefix: str) -> bool:
        """True if any recorded call starts with *prefix*."""
        return any(c[: len(prefix)] == list(prefix) for c, _ in self.calls)


@pytest.fixture
def fake_git(monkeypatch):
    fg = FakeGit()
    monkeypatch.setattr(vault_sync, "_run_git", fg)
    return fg


def make_checkout(base: pathlib.Path, name: str, head: str = "h1",
                  fake_git: FakeGit | None = None) -> pathlib.Path:
    """Create <base>/<name>/.git so ensure_cloned treats it as present.

    With *fake_git*, also registers *head* as the checkout's current HEAD.
    """
    vdir = base / name
    (vdir / ".git").mkdir(parents=True)
    if fake_git is not None:
        fake_git.heads[str(vdir)] = head
    return vdir


# ──────────────────────── parse_vaults ────────────────────────

class TestParseVaults:
    def test_multiple_entries(self):
        got = vault_sync.parse_vaults("humblewood=https://a.git; dnd_handbook=https://b.git")
        assert got == {"humblewood": "https://a.git", "dnd_handbook": "https://b.git"}

    def test_empty_and_malformed_skipped(self):
        got = vault_sync.parse_vaults(";;ok=https://x.git;noequals;;=https://y.git")
        assert got == {"ok": "https://x.git"}

    def test_unsafe_names_rejected(self):
        got = vault_sync.parse_vaults("a/b=https://x.git;..=https://y.git;c\\d=https://z.git")
        assert got == {}

    def test_empty_string(self):
        assert vault_sync.parse_vaults("") == {}


# ──────────────────────── ensure_cloned ────────────────────────

class TestEnsureCloned:
    async def test_existing_checkout_no_clone(self, tmp_path, fake_git):
        vdir = make_checkout(tmp_path, "v")
        assert await vault_sync.ensure_cloned("https://x.git", vdir) is None
        assert not fake_git.ran("clone")

    async def test_missing_dir_clones(self, tmp_path, fake_git):
        err = await vault_sync.ensure_cloned("https://x.git", tmp_path / "v")
        assert err is None
        assert fake_git.ran("clone")

    async def test_two_way_sets_identity(self, tmp_path, fake_git):
        await vault_sync.ensure_cloned("https://x.git", tmp_path / "v", two_way=True)
        assert fake_git.ran("config", "user.name")
        assert fake_git.ran("config", "user.email")

    async def test_one_way_no_identity(self, tmp_path, fake_git):
        await vault_sync.ensure_cloned("https://x.git", tmp_path / "v")
        assert not fake_git.ran("config")

    async def test_clone_failure_returns_error(self, tmp_path, monkeypatch):
        def boom(args, cwd):
            return (128, "fatal: could not read Username")
        monkeypatch.setattr(vault_sync, "_run_git", boom)
        err = await vault_sync.ensure_cloned("https://x.git", tmp_path / "v")
        assert err is not None and "could not read Username" in err


# ──────────────────────── mirror_pull (one-way) ────────────────────────

class TestMirrorPull:
    async def test_changed_head(self, tmp_path, fake_git):
        vdir = make_checkout(tmp_path, "v", head="a1", fake_git=fake_git)
        fake_git.remote_head = "b2"
        ok, changed, err = await vault_sync.mirror_pull(vdir)
        assert (ok, changed, err) == (True, True, None)
        assert fake_git.ran("reset", "--hard")

    async def test_unchanged_head(self, tmp_path, fake_git):
        vdir = make_checkout(tmp_path, "v", head="same", fake_git=fake_git)
        fake_git.remote_head = "same"
        ok, changed, err = await vault_sync.mirror_pull(vdir)
        assert (ok, changed, err) == (True, False, None)

    async def test_fetch_failure(self, tmp_path, fake_git):
        vdir = make_checkout(tmp_path, "v", head="a1", fake_git=fake_git)
        fake_git.fail_fetch = True
        ok, changed, err = await vault_sync.mirror_pull(vdir)
        assert ok is False and changed is False
        assert "fetch failed" in (err or "")


# ──────────────────────── campaign_sync (two-way) ────────────────────────

class TestCampaignSync:
    async def test_dirty_commits_and_pushes(self, tmp_path, fake_git):
        vdir = make_checkout(tmp_path, "v", head="a1", fake_git=fake_git)
        fake_git.dirty.add(str(vdir))
        ok, changed, err = await vault_sync.campaign_sync(vdir)
        assert (ok, changed, err) == (True, True, None)
        assert fake_git.ran("add", "-A")
        assert fake_git.ran("commit")
        assert fake_git.ran("push")

    async def test_clean_no_commit(self, tmp_path, fake_git):
        vdir = make_checkout(tmp_path, "v", head="a1", fake_git=fake_git)
        fake_git.remote_head = "a1"
        ok, changed, err = await vault_sync.campaign_sync(vdir)
        assert (ok, changed, err) == (True, False, None)
        assert not fake_git.ran("commit")
        assert not fake_git.ran("push")

    async def test_pull_only_remote_change(self, tmp_path, fake_git):
        vdir = make_checkout(tmp_path, "v", head="a1", fake_git=fake_git)
        fake_git.remote_head = "b2"  # user pushed an Obsidian edit elsewhere
        ok, changed, err = await vault_sync.campaign_sync(vdir)
        assert (ok, changed, err) == (True, True, None)
        assert not fake_git.ran("commit")

    async def test_push_rejected_once_then_retry(self, tmp_path, fake_git):
        vdir = make_checkout(tmp_path, "v", head="a1", fake_git=fake_git)
        fake_git.dirty.add(str(vdir))
        fake_git.push_reject_once = True
        ok, changed, err = await vault_sync.campaign_sync(vdir)
        assert (ok, changed, err) == (True, True, None)
        pushes = [c for c, _ in fake_git.calls if c[0] == "push"]
        assert len(pushes) == 2  # rejected once, retried after rebase

    async def test_pull_conflict_aborts_and_keeps_local(self, tmp_path, fake_git):
        vdir = make_checkout(tmp_path, "v", head="a1", fake_git=fake_git)
        fake_git.conflict_pull = True
        ok, changed, err = await vault_sync.campaign_sync(vdir)
        assert ok is False and changed is False
        assert "conflict" in (err or "")
        assert fake_git.ran("rebase", "--abort")
        assert not fake_git.ran("push")

    async def test_push_conflict_aborts_keeps_commit(self, tmp_path, fake_git):
        vdir = make_checkout(tmp_path, "v", head="a1", fake_git=fake_git)
        fake_git.dirty.add(str(vdir))
        # Sequence: pull#1 OK → commit → push rejected (always) → retry
        # pull#2 hits a rebase conflict → abort, local commit kept.
        state = {"pulls": 0}

        def flaky(args, cwd):
            if args[0] == "push":
                return (1, "! [rejected] main -> main (non-fast-forward)")
            if args[0] == "pull":
                state["pulls"] += 1
                if state["pulls"] >= 2:
                    return (1, "CONFLICT (content): Merge conflict in notes.md")
            return fake_git(args, cwd)

        with patch.object(vault_sync, "_run_git", side_effect=flaky):
            ok, changed, err = await vault_sync.campaign_sync(vdir)
        assert ok is False and changed is True  # local commit kept → re-index still needed
        assert "conflict" in (err or "")
        assert fake_git.ran("rebase", "--abort")


# ──────────────────────── start / loop ────────────────────────

class TestStartAndLoop:
    @staticmethod
    def _closing_spawn(coro, **kwargs):
        coro.close()  # avoid "coroutine never awaited" warnings

    async def test_disabled_when_no_vaults(self, monkeypatch):
        import config.settings as settings
        monkeypatch.setattr(settings, "OBSIDIAN_VAULTS", "")
        with patch("utils.background_tasks.spawn_tracked_task",
                   side_effect=self._closing_spawn) as spawn:
            vault_sync.start_vault_sync()
        spawn.assert_not_called()

    async def test_enabled_spawns_task(self, monkeypatch):
        import config.settings as settings
        monkeypatch.setattr(settings, "OBSIDIAN_VAULTS", "v=https://x.git")
        with patch("utils.background_tasks.spawn_tracked_task",
                   side_effect=self._closing_spawn) as spawn:
            vault_sync.start_vault_sync()
        spawn.assert_called_once()

    async def test_loop_reindexes_once_when_any_vault_changed(
        self, tmp_path, monkeypatch, fake_git
    ):
        import config.settings as settings
        kb = tmp_path / "kb"
        kb.mkdir()
        v1, v2 = make_checkout(kb, "v1", head="a"), make_checkout(kb, "v2", head="b")
        fake_git.heads[str(v1)] = "a"
        fake_git.heads[str(v2)] = "b"
        fake_git.remote_head = "z"  # both vaults have new remote commits

        monkeypatch.setattr(settings, "OBSIDIAN_VAULTS", "v1=https://x.git;v2=https://y.git")
        monkeypatch.setattr(settings, "OBSIDIAN_CAMPAIGN_VAULT", "")
        monkeypatch.setattr(settings, "OBSIDIAN_VAULT_PULL_INTERVAL", 0)  # single pass
        monkeypatch.setattr(settings, "KB_PATH", kb)

        from kb import retrievers
        sync_mock = AsyncMock(return_value=(None, {"changed_count": 2}))
        monkeypatch.setattr(retrievers, "sync_kb_store", sync_mock)

        await vault_sync.vault_sync_loop()

        sync_mock.assert_awaited_once()  # once per tick, not per vault

    async def test_loop_failing_vault_does_not_block_others(
        self, tmp_path, monkeypatch, fake_git
    ):
        import config.settings as settings
        kb = tmp_path / "kb"
        kb.mkdir()
        v1, v2 = make_checkout(kb, "bad", head="a"), make_checkout(kb, "good", head="b")
        fake_git.remote_head = "z"

        monkeypatch.setattr(settings, "OBSIDIAN_VAULTS", "bad=https://x.git;good=https://y.git")
        monkeypatch.setattr(settings, "OBSIDIAN_CAMPAIGN_VAULT", "")
        monkeypatch.setattr(settings, "OBSIDIAN_VAULT_PULL_INTERVAL", 0)
        monkeypatch.setattr(settings, "KB_PATH", kb)

        real = fake_git

        def flaky(args, cwd):
            if str(cwd).endswith("bad") and args[0] == "fetch":
                return (1, "fatal: auth failed")
            return real(args, cwd)

        monkeypatch.setattr(vault_sync, "_run_git", flaky)

        from kb import retrievers
        sync_mock = AsyncMock(return_value=(None, {"changed_count": 1}))
        monkeypatch.setattr(retrievers, "sync_kb_store", sync_mock)

        await vault_sync.vault_sync_loop()  # must not raise

        sync_mock.assert_awaited_once()  # 'good' still re-indexed

    async def test_loop_no_changes_no_reindex(self, tmp_path, monkeypatch, fake_git):
        import config.settings as settings
        kb = tmp_path / "kb"
        kb.mkdir()
        v1 = make_checkout(kb, "v1", head="same", fake_git=fake_git)
        fake_git.remote_head = "same"

        monkeypatch.setattr(settings, "OBSIDIAN_VAULTS", "v1=https://x.git")
        monkeypatch.setattr(settings, "OBSIDIAN_CAMPAIGN_VAULT", "")
        monkeypatch.setattr(settings, "OBSIDIAN_VAULT_PULL_INTERVAL", 0)
        monkeypatch.setattr(settings, "KB_PATH", kb)

        from kb import retrievers
        sync_mock = AsyncMock()
        monkeypatch.setattr(retrievers, "sync_kb_store", sync_mock)

        await vault_sync.vault_sync_loop()
        sync_mock.assert_not_awaited()

    async def test_unknown_campaign_name_warns_and_stays_oneway(
        self, tmp_path, monkeypatch, fake_git, caplog
    ):
        import config.settings as settings
        kb = tmp_path / "kb"
        kb.mkdir()
        v1 = make_checkout(kb, "v1", head="same", fake_git=fake_git)
        fake_git.remote_head = "same"

        monkeypatch.setattr(settings, "OBSIDIAN_VAULTS", "v1=https://x.git")
        monkeypatch.setattr(settings, "OBSIDIAN_CAMPAIGN_VAULT", "nope")
        monkeypatch.setattr(settings, "OBSIDIAN_VAULT_PULL_INTERVAL", 0)
        monkeypatch.setattr(settings, "KB_PATH", kb)

        with caplog.at_level("WARNING", logger="bot.vault_sync"):
            await vault_sync.vault_sync_loop()

        assert any("not in OBSIDIAN_VAULTS" in m for m in caplog.messages)
        assert not fake_git.ran("push")  # treated as one-way mirror
