"""
Tests for dataset loading and formatting.

These tests verify that MBPP is correctly formatted for GRPOTrainer.
Requires internet access to download the dataset on first run.
Requires the `datasets` library (installed on GPU instance, not locally).
"""

import pytest

try:
    from src.dataset import load_mbpp_for_grpo, SYSTEM_PROMPT
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

pytestmark = pytest.mark.skipif(not HAS_DATASETS, reason="datasets library not installed")


class TestMBPPDataset:
    @pytest.fixture(scope="class")
    def dataset(self):
        """Load the MBPP training split once for all tests."""
        return load_mbpp_for_grpo(split="train")

    def test_dataset_loads(self, dataset):
        """Check that we get a non-empty dataset."""
        assert len(dataset) > 0

    def test_has_required_columns(self, dataset):
        """GRPOTrainer requires 'prompt'; we also need test metadata."""
        assert "prompt" in dataset.column_names
        assert "test_list" in dataset.column_names
        assert "test_setup_code" in dataset.column_names

    def test_prompt_format(self, dataset):
        """Prompts should be in OpenAI conversation format."""
        row = dataset[0]
        prompt = row["prompt"]
        assert isinstance(prompt, list)
        assert len(prompt) == 2
        assert prompt[0]["role"] == "system"
        assert prompt[0]["content"] == SYSTEM_PROMPT
        assert prompt[1]["role"] == "user"
        assert len(prompt[1]["content"]) > 0

    def test_test_list_non_empty(self, dataset):
        """Each problem should have at least one test assertion."""
        for i in range(min(10, len(dataset))):
            assert len(dataset[i]["test_list"]) > 0

    def test_tests_are_assertions(self, dataset):
        """Tests should be assert statements."""
        row = dataset[0]
        for test in row["test_list"]:
            assert "assert" in test.lower() or "==" in test or "raise" in test.lower()
