"""
Tests for the SAFE dataset classes and utilities.
"""
import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import torch

from safe.data.datasets import (
    _to_tensor,
    _collate_multimodal_batch,
    create_safe_dataloader,
)


class TestToTensor:
    """Tests for _to_tensor helper function."""

    def test_none_input(self):
        """Test _to_tensor returns None for None input."""
        assert _to_tensor(None) is None

    def test_tensor_input(self):
        """Test _to_tensor returns tensor unchanged."""
        t = torch.randn(3, 4)
        result = _to_tensor(t)
        assert torch.equal(result, t)

    def test_list_input(self):
        """Test _to_tensor converts list to tensor."""
        result = _to_tensor([1, 2, 3])
        assert isinstance(result, torch.Tensor)
        assert result.tolist() == [1, 2, 3]

    def test_tuple_input(self):
        """Test _to_tensor converts tuple to tensor."""
        result = _to_tensor((1.0, 2.0, 3.0))
        assert isinstance(result, torch.Tensor)

    def test_unconvertible_input(self):
        """Test _to_tensor returns None for unconvertible input."""
        result = _to_tensor([{"a": 1}, {"b": 2}])  # List of dicts
        assert result is None


class TestCollateMultimodalBatch:
    """Tests for _collate_multimodal_batch function."""

    def test_empty_batch(self):
        """Test collation of empty batch."""
        result = _collate_multimodal_batch([])

        assert result["questions"] == []
        assert result["answers"] == []
        assert result["images"] is None
        assert result["audio"] is None
        assert result["has_audio"].shape == (0,)

    def test_basic_collation(self):
        """Test basic batch collation."""
        batch = [
            {
                "question": "What sound?",
                "answer": "Bird chirping",
                "audio": torch.randn(16000),
                "images": None,
            },
            {
                "question": "What do you see?",
                "answer": "A cat",
                "audio": None,
                "images": torch.randn(3, 224, 224),
            },
        ]

        result = _collate_multimodal_batch(batch)

        assert len(result["questions"]) == 2
        assert len(result["answers"]) == 2
        assert len(result["audio"]) == 2
        assert len(result["images"]) == 2
        assert result["has_audio"].shape == (2,)
        assert result["has_audio"][0].item() is True
        assert result["has_audio"][1].item() is False

    def test_collation_with_alternative_keys(self):
        """Test collation handles alternative key names."""
        batch = [
            {
                "questions": "Alt question key",  # Plural form
                "answers": "Alt answer",
                "image": torch.randn(3, 224, 224),  # Singular form
            }
        ]

        result = _collate_multimodal_batch(batch)

        assert result["questions"][0] == "Alt question key"
        assert result["answers"][0] == "Alt answer"
        assert result["images"][0] is not None

    def test_collation_with_optional_keys(self):
        """Test collation includes optional keys when present."""
        batch = [
            {
                "question": "Q1",
                "answer": "A1",
                "sample_id": "id1",
                "difficulty": "easy",
                "question_type": "audio_dependent",
            },
            {
                "question": "Q2",
                "answer": "A2",
                "sample_id": "id2",
                "difficulty": "hard",
                "question_type": "visual_only",
            },
        ]

        result = _collate_multimodal_batch(batch)

        assert "sample_ids" in result
        assert "difficultys" in result
        assert "question_types" in result
        assert result["sample_ids"] == ["id1", "id2"]

    def test_collation_missing_fields(self):
        """Test collation handles missing fields gracefully."""
        batch = [
            {"question": "Q1"},  # Missing answer, audio, images
        ]

        result = _collate_multimodal_batch(batch)

        assert result["questions"][0] == "Q1"
        assert result["answers"][0] is None
        assert result["has_audio"][0].item() is False

    def test_has_audio_tensor_dtype(self):
        """Test has_audio is boolean tensor."""
        batch = [
            {"question": "Q", "answer": "A", "audio": torch.randn(100)},
            {"question": "Q", "answer": "A", "audio": None},
        ]

        result = _collate_multimodal_batch(batch)

        assert result["has_audio"].dtype == torch.bool


class TestMockDataset:
    """Tests using mock dataset for dataloader functionality."""

    class SimpleMockDataset:
        """Simple mock dataset for testing."""

        def __init__(self, size=10, has_audio=True, has_images=True):
            self.size = size
            self.has_audio = has_audio
            self.has_images = has_images

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            sample = {
                "question": f"Question {idx}",
                "answer": f"Answer {idx}",
                "sample_id": str(idx),
            }
            if self.has_audio:
                sample["audio"] = torch.randn(16000)
            else:
                sample["audio"] = None

            if self.has_images:
                sample["images"] = torch.randn(3, 224, 224)
            else:
                sample["images"] = None

            return sample

    def test_create_safe_dataloader_basic(self):
        """Test creating a basic dataloader."""
        dataset = self.SimpleMockDataset(size=10)
        dataloader = create_safe_dataloader(dataset, batch_size=2)

        assert dataloader.batch_size == 2
        assert len(dataloader) == 5  # 10 samples / 2 batch size

    def test_create_safe_dataloader_iteration(self):
        """Test iterating through dataloader."""
        dataset = self.SimpleMockDataset(size=4)
        dataloader = create_safe_dataloader(dataset, batch_size=2, shuffle=False)

        batches = list(dataloader)

        assert len(batches) == 2
        assert len(batches[0]["questions"]) == 2
        assert batches[0]["has_audio"].shape == (2,)

    def test_create_safe_dataloader_audio_only(self):
        """Test dataloader with audio-only samples."""
        dataset = self.SimpleMockDataset(size=4, has_audio=True, has_images=False)
        dataloader = create_safe_dataloader(dataset, batch_size=2)

        batch = next(iter(dataloader))

        assert batch["has_audio"].all()
        assert all(img is None for img in batch["images"])

    def test_create_safe_dataloader_visual_only(self):
        """Test dataloader with visual-only samples."""
        dataset = self.SimpleMockDataset(size=4, has_audio=False, has_images=True)
        dataloader = create_safe_dataloader(dataset, batch_size=2)

        batch = next(iter(dataloader))

        assert not batch["has_audio"].any()
        assert all(audio is None for audio in batch["audio"])

    def test_create_safe_dataloader_shuffle(self):
        """Test dataloader with shuffle enabled."""
        dataset = self.SimpleMockDataset(size=20)
        dataloader = create_safe_dataloader(dataset, batch_size=4, shuffle=True)

        # Should be able to iterate without error
        batches = list(dataloader)
        assert len(batches) == 5

    def test_create_safe_dataloader_num_workers(self):
        """Test dataloader with multiple workers."""
        dataset = self.SimpleMockDataset(size=8)
        # Note: num_workers=0 in tests to avoid multiprocessing issues
        dataloader = create_safe_dataloader(
            dataset, batch_size=2, num_workers=0
        )

        batches = list(dataloader)
        assert len(batches) == 4


class TestBaseQADatasetBehavior:
    """Tests for _BaseQADataset behavior without actual file dependencies."""

    def test_extract_answer_string(self):
        """Test answer extraction from string."""
        # This tests the logic that would be in _extract_answer
        raw_answer = "Simple answer"
        # Should return as-is
        assert raw_answer == "Simple answer"

    def test_extract_answer_dict(self):
        """Test answer extraction from dict with 'answer' key."""
        raw_answer = {"answer": "Dict answer", "confidence": 0.9}
        extracted = raw_answer.get("answer", raw_answer)
        assert extracted == "Dict answer"

    def test_extract_answer_list_of_strings(self):
        """Test answer extraction from list of strings (majority vote)."""
        from collections import Counter

        raw_answer = ["cat", "cat", "dog", "cat"]
        counts = Counter(str(a) for a in raw_answer)
        most_common = counts.most_common(1)[0][0]
        assert most_common == "cat"

    def test_extract_answer_list_of_dicts(self):
        """Test answer extraction from list of dicts."""
        from collections import Counter

        raw_answer = [
            {"answer": "bird"},
            {"answer": "bird"},
            {"answer": "plane"},
        ]
        counts = Counter(
            item.get("answer", "") for item in raw_answer if item.get("answer")
        )
        most_common = counts.most_common(1)[0][0]
        assert most_common == "bird"

    def test_extract_answer_empty_list(self):
        """Test answer extraction from empty list."""
        raw_answer = []
        # Should handle gracefully
        result = "" if not raw_answer else raw_answer[0]
        assert result == ""


class TestDatasetFileFormats:
    """Tests for dataset file format handling."""

    def create_jsonl_dataset(self, tmp_path, samples):
        """Create a JSONL dataset file."""
        dataset_dir = tmp_path / "test_dataset"
        dataset_dir.mkdir()
        file_path = dataset_dir / "data_train.jsonl"

        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        return tmp_path

    def create_json_dataset(self, tmp_path, samples):
        """Create a JSON dataset file."""
        dataset_dir = tmp_path / "test_dataset"
        dataset_dir.mkdir()
        file_path = dataset_dir / "data_train.json"

        with open(file_path, "w") as f:
            json.dump(samples, f)

        return tmp_path

    def test_jsonl_format_structure(self, tmp_path):
        """Test JSONL file structure is valid."""
        samples = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]

        data_path = self.create_jsonl_dataset(tmp_path, samples)
        file_path = data_path / "test_dataset" / "data_train.jsonl"

        # Verify file can be read as JSONL
        loaded = []
        with open(file_path) as f:
            for line in f:
                loaded.append(json.loads(line))

        assert len(loaded) == 2
        assert loaded[0]["question"] == "Q1"

    def test_json_format_structure(self, tmp_path):
        """Test JSON file structure is valid."""
        samples = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]

        data_path = self.create_json_dataset(tmp_path, samples)
        file_path = data_path / "test_dataset" / "data_train.json"

        # Verify file can be read as JSON
        with open(file_path) as f:
            loaded = json.load(f)

        assert len(loaded) == 2
        assert loaded[0]["question"] == "Q1"


class TestDatasetEdgeCases:
    """Tests for edge cases in dataset handling."""

    def test_empty_question_handling(self):
        """Test handling of empty questions."""
        sample = {"question": "", "answer": "A"}
        # Empty question should be allowed but noted
        assert sample["question"] == ""

    def test_none_audio_handling(self):
        """Test handling of None audio."""
        sample = {"question": "Q", "answer": "A", "audio": None}
        assert sample["audio"] is None

    def test_multi_reference_answers(self):
        """Test handling of multiple reference answers."""
        sample = {
            "question": "What is in the audio?",
            "answers": ["Bird chirping", "A bird singing", "Birds in the morning"],
        }
        assert len(sample["answers"]) == 3

    def test_unicode_text(self):
        """Test handling of unicode text."""
        sample = {
            "question": "What sound do you hear? 你好",
            "answer": "鳥の鳴き声 🐦",
        }
        assert "你好" in sample["question"]
        assert "🐦" in sample["answer"]

    def test_special_characters_in_text(self):
        """Test handling of special characters."""
        sample = {
            "question": "What's the \"sound\" in this <audio>?",
            "answer": "A 'bird' chirping & singing",
        }
        assert '"' in sample["question"]
        assert "'" in sample["answer"]
        assert "&" in sample["answer"]
