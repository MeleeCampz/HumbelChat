"""utils.studio_model — shared repo:quant slug helper for Studio loads."""
from __future__ import annotations

from utils.studio_model import is_resident, load_payload, split_repo_variant


class TestSplitRepoVariant:
    def test_splits_repo_and_quant(self):
        assert split_repo_variant(
            "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-NVFP4_50-Q4K_M"
        ) == ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-NVFP4_50-Q4K_M")

    def test_plain_repo_has_empty_variant(self):
        assert split_repo_variant("unsloth/Qwen-Image-2.1-Turbo-GGUF") == (
            "unsloth/Qwen-Image-2.1-Turbo-GGUF", ""
        )

    def test_splits_at_first_colon_only(self):
        # Quant slugs never contain ':', but a weird one must not lose data.
        assert split_repo_variant("repo:variant:extra") == ("repo", "variant:extra")

    def test_empty_slug(self):
        assert split_repo_variant("") == ("", "")
        assert split_repo_variant(None) == ("", "")  # type: ignore[arg-type]


class TestLoadPayload:
    def test_repo_quant_split(self):
        payload = load_payload("unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-NVFP4_50-Q4K_M")
        assert payload == {
            "model_path": "unsloth/Qwen3.8-27B-GGUF",
            "gguf_variant": "Qwen3.8-27B-NVFP4_50-Q4K_M",
        }

    def test_plain_repo_has_no_variant_key(self):
        payload = load_payload("unsloth/Qwen-Image-2.1-Turbo-GGUF")
        assert payload == {"model_path": "unsloth/Qwen-Image-2.1-Turbo-GGUF"}
        assert "gguf_variant" not in payload

    def test_extra_merged_not_mutated(self):
        extra = {"gpu_ids": [1]}
        payload = load_payload("repo:v", extra)
        assert payload == {"model_path": "repo", "gguf_variant": "v", "gpu_ids": [1]}
        assert extra == {"gpu_ids": [1]}  # unchanged


class TestIsResident:
    def test_bare_id_plus_matching_quant(self):
        # Studio reports the id WITHOUT the quant, in a separate field.
        m = {"id": "unsloth/Qwen3.8-27B-GGUF", "quant": "Qwen3.8-27B-NVFP4_50-Q4K_M",
             "loaded": True}
        assert is_resident(m, "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-NVFP4_50-Q4K_M")

    def test_wrong_quant_is_not_resident(self):
        m = {"id": "unsloth/Qwen3.8-27B-GGUF", "quant": "Other-Q4_K_M", "loaded": True}
        assert not is_resident(m, "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-NVFP4_50-Q4K_M")

    def test_not_loaded_flag_is_not_resident(self):
        m = {"id": "unsloth/Qwen3.8-27B-GGUF", "quant": "Qwen3.8-27B-NVFP4_50-Q4K_M",
             "loaded": False}
        assert not is_resident(m, "unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-NVFP4_50-Q4K_M")

    def test_missing_quant_counts_as_match(self):
        m = {"id": "repo/x", "loaded": True}
        assert is_resident(m, "repo/x:whatever")

    def test_plain_slug_matches_bare_id(self):
        m = {"id": "unsloth/Qwen-Image-2.1-Turbo-GGUF", "loaded": True}
        assert is_resident(m, "unsloth/Qwen-Image-2.1-Turbo-GGUF")
