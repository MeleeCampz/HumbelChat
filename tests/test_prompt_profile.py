"""The campaign profile file (docs/campaign-profile.md): where it is found, and what a
missing one does. The composer deliberately holds NO fallback campaign wording, so a lost
profile must announce itself rather than quietly paint without the campaign's sentences.
"""
from __future__ import annotations

from bot_core import prompt_profile


def test_a_lost_profile_file_says_so(monkeypatch, tmp_path):
    """A deployment without a profile is legal; a deployment that LOST one is not silent.

    The profile lives under `data/`, which is gitignored: a recreated checkout or a second
    machine loses it, and every prompt would stop opening with the campaign's sentence with
    nothing in the log saying why.
    """
    monkeypatch.setenv("IMAGE_PROMPT_PROFILE", str(tmp_path / "absent.md"))
    p = prompt_profile.load()
    assert p.source == "<defaults>" and p.style == "" and p.world_rule == ""
    assert any("no campaign profile file found" in w for w in p.warnings), p.warnings
    assert any("absent.md" in w for w in p.warnings), \
        "the warning must name where it looked, not just that it found nothing"


def test_a_profile_inside_a_vault_folder_is_found_too(monkeypatch, tmp_path):
    """`<KB_PATH>/Campaign Prompt Profile.md`, then `<KB_PATH>/<vault>/…`.

    The second search path is what lets the campaign wording live IN the vault repo instead
    of in the host's gitignored `data/`: it travels with the campaign it describes,
    `vault_sync` carries it, and no code has to know. Paths written inside the file
    (`party_sheet`) resolve against KB_PATH either way, so moving it one level down changes
    nothing — and the KB-root copy still wins when both exist.
    """
    monkeypatch.delenv("IMAGE_PROMPT_PROFILE", raising=False)
    kb = tmp_path / "knowledge"
    (kb / "myvault").mkdir(parents=True)
    (kb / "myvault" / "Campaign Prompt Profile.md").write_text(
        "> style: ZZZ_VAULT_STYLE\n> world_rule: ZZZ_VAULT_RULE\n", encoding="utf-8")
    monkeypatch.setattr(prompt_profile.settings, "KB_PATH", str(kb))

    p = prompt_profile.load()
    assert p.style == "ZZZ_VAULT_STYLE" and p.world_rule == "ZZZ_VAULT_RULE", p.source
    assert "myvault" in p.source and not p.warnings, p.warnings

    (kb / "Campaign Prompt Profile.md").write_text("> style: ZZZ_ROOT_STYLE\n",
                                                   encoding="utf-8")
    assert prompt_profile.load().style == "ZZZ_ROOT_STYLE", "the most specific path first"
