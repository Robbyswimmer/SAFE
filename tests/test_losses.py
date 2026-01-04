"""
Tests for the SAFE training loss functions.
"""
import pytest
import torch
import torch.nn as nn
from typing import Dict

from safe.training.losses import (
    RetentionLoss,
    AudioTaskLoss,
    CombinedStageLoss,
    RewardFunction,
    ConstrainedRetentionLoss,
)


class TestRetentionLoss:
    """Tests for RetentionLoss class."""

    def test_initialization_default_params(self):
        """Test RetentionLoss initializes with default parameters."""
        loss_fn = RetentionLoss()
        assert loss_fn.distillation_weight == 1.0
        assert loss_fn.fisher_weight == 0.1
        assert loss_fn.temperature == 2.0
        assert loss_fn.use_fisher_information is True
        assert loss_fn.fisher_information is None

    def test_initialization_custom_params(self):
        """Test RetentionLoss with custom parameters."""
        loss_fn = RetentionLoss(
            distillation_weight=0.5,
            fisher_weight=0.2,
            temperature=3.0,
            use_fisher_information=False,
        )
        assert loss_fn.distillation_weight == 0.5
        assert loss_fn.fisher_weight == 0.2
        assert loss_fn.temperature == 3.0
        assert loss_fn.use_fisher_information is False

    def test_set_fisher_information(self):
        """Test setting Fisher information matrix."""
        loss_fn = RetentionLoss()
        fisher_info = {"param1": torch.randn(10, 10), "param2": torch.randn(5)}
        loss_fn.set_fisher_information(fisher_info)
        assert loss_fn.fisher_information is not None
        assert "param1" in loss_fn.fisher_information

    def test_kl_divergence_loss_same_shape(self):
        """Test KL divergence with matching tensor shapes."""
        loss_fn = RetentionLoss()
        student_logits = torch.randn(2, 10, 100)
        teacher_logits = torch.randn(2, 10, 100)

        kl_loss = loss_fn.kl_divergence_loss(student_logits, teacher_logits)

        assert kl_loss.dim() == 0  # Scalar
        assert not torch.isnan(kl_loss)
        assert not torch.isinf(kl_loss)
        assert kl_loss >= 0  # KL divergence is non-negative

    def test_kl_divergence_loss_seq_len_mismatch(self):
        """Test KL divergence handles sequence length mismatch."""
        loss_fn = RetentionLoss()
        student_logits = torch.randn(2, 15, 100)  # Longer sequence
        teacher_logits = torch.randn(2, 10, 100)

        kl_loss = loss_fn.kl_divergence_loss(student_logits, teacher_logits)

        assert kl_loss.dim() == 0
        assert not torch.isnan(kl_loss)

    def test_kl_divergence_loss_vocab_size_mismatch(self):
        """Test KL divergence handles vocab size mismatch."""
        loss_fn = RetentionLoss()
        student_logits = torch.randn(2, 10, 120)  # Larger vocab
        teacher_logits = torch.randn(2, 10, 100)

        kl_loss = loss_fn.kl_divergence_loss(student_logits, teacher_logits)

        assert kl_loss.dim() == 0
        assert not torch.isnan(kl_loss)

    def test_kl_divergence_loss_batch_size_mismatch_expand(self):
        """Test KL divergence expands batch size when one is 1."""
        loss_fn = RetentionLoss()
        student_logits = torch.randn(4, 10, 100)
        teacher_logits = torch.randn(1, 10, 100)  # Batch size 1

        kl_loss = loss_fn.kl_divergence_loss(student_logits, teacher_logits)

        assert kl_loss.dim() == 0
        assert not torch.isnan(kl_loss)

    def test_kl_divergence_temperature_scaling(self):
        """Test that temperature scaling affects the loss."""
        student_logits = torch.randn(2, 10, 100)
        teacher_logits = torch.randn(2, 10, 100)

        loss_fn_t1 = RetentionLoss(temperature=1.0)
        loss_fn_t4 = RetentionLoss(temperature=4.0)

        kl_loss_t1 = loss_fn_t1.kl_divergence_loss(student_logits, teacher_logits)
        kl_loss_t4 = loss_fn_t4.kl_divergence_loss(student_logits, teacher_logits)

        # Different temperatures should give different losses
        assert not torch.isclose(kl_loss_t1, kl_loss_t4)

    def test_fisher_regularization_loss_no_fisher_info(self):
        """Test Fisher regularization falls back to L2 without Fisher info."""
        loss_fn = RetentionLoss()
        current_params = {"layer.weight": torch.randn(10, 10)}
        base_params = {"layer.weight": torch.randn(10, 10)}

        reg_loss = loss_fn.fisher_regularization_loss(current_params, base_params)

        assert reg_loss.dim() == 0
        assert not torch.isnan(reg_loss)
        assert reg_loss >= 0

    def test_fisher_regularization_loss_with_fisher_info(self):
        """Test Fisher regularization with precomputed Fisher information."""
        loss_fn = RetentionLoss()
        current_params = {"layer.weight": torch.randn(10, 10)}
        base_params = {"layer.weight": torch.randn(10, 10)}
        fisher_info = {"layer.weight": torch.abs(torch.randn(10, 10))}

        loss_fn.set_fisher_information(fisher_info)
        reg_loss = loss_fn.fisher_regularization_loss(current_params, base_params)

        assert reg_loss.dim() == 0
        assert not torch.isnan(reg_loss)
        assert reg_loss >= 0

    def test_fisher_regularization_loss_empty_params(self):
        """Test Fisher regularization with empty parameters."""
        loss_fn = RetentionLoss()
        reg_loss = loss_fn.fisher_regularization_loss({}, {})
        assert reg_loss.item() == 0.0

    def test_fisher_regularization_zero_when_params_equal(self):
        """Test Fisher regularization is zero when params are identical."""
        loss_fn = RetentionLoss()
        params = {"layer.weight": torch.randn(10, 10)}

        reg_loss = loss_fn.fisher_regularization_loss(params, params)

        assert reg_loss.item() == 0.0

    def test_forward_basic(self):
        """Test forward pass computes all loss components."""
        loss_fn = RetentionLoss()
        safe_logits = torch.randn(2, 10, 100)
        base_logits = torch.randn(2, 10, 100)

        result = loss_fn(safe_logits, base_logits)

        assert "retention_loss" in result
        assert "distillation_loss" in result
        assert "fisher_loss" in result
        assert not torch.isnan(result["retention_loss"])
        assert not torch.isnan(result["distillation_loss"])

    def test_forward_with_model_params(self):
        """Test forward with model parameters for Fisher regularization."""
        loss_fn = RetentionLoss()
        safe_logits = torch.randn(2, 10, 100)
        base_logits = torch.randn(2, 10, 100)
        safe_params = {"layer.weight": torch.randn(10, 10, requires_grad=True)}
        base_params = {"layer.weight": torch.randn(10, 10)}

        result = loss_fn(
            safe_logits,
            base_logits,
            safe_model_params=safe_params,
            base_model_params=base_params,
        )

        assert result["fisher_loss"] >= 0

    def test_forward_gradient_flow(self):
        """Test that gradients flow through the loss."""
        loss_fn = RetentionLoss()
        safe_logits = torch.randn(2, 10, 100, requires_grad=True)
        base_logits = torch.randn(2, 10, 100)

        result = loss_fn(safe_logits, base_logits)
        result["retention_loss"].backward()

        assert safe_logits.grad is not None
        assert torch.isfinite(safe_logits.grad).all()


class TestAudioTaskLoss:
    """Tests for AudioTaskLoss class."""

    def test_initialization_qa_task(self):
        """Test AudioTaskLoss initializes for QA task."""
        loss_fn = AudioTaskLoss(task_type="qa")
        assert loss_fn.task_type == "qa"
        assert loss_fn.label_smoothing == 0.1

    def test_initialization_caption_task(self):
        """Test AudioTaskLoss initializes for caption task."""
        loss_fn = AudioTaskLoss(task_type="caption", label_smoothing=0.0)
        assert loss_fn.task_type == "caption"
        assert loss_fn.label_smoothing == 0.0

    def test_initialization_invalid_task_raises(self):
        """Test AudioTaskLoss raises for invalid task type."""
        with pytest.raises(ValueError, match="Unsupported task type"):
            AudioTaskLoss(task_type="invalid")

    def test_set_debug(self):
        """Test debug mode can be toggled."""
        loss_fn = AudioTaskLoss()
        assert loss_fn.debug is False

        loss_fn.set_debug(True)
        assert loss_fn.debug is True

        loss_fn.set_debug(False)
        assert loss_fn.debug is False

    def test_forward_basic_qa(self):
        """Test basic forward pass for QA task."""
        loss_fn = AudioTaskLoss(task_type="qa")
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))

        loss = loss_fn(logits, labels)

        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0

    def test_forward_with_attention_mask(self):
        """Test forward pass with attention mask."""
        loss_fn = AudioTaskLoss(task_type="qa")
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        attention_mask = torch.ones(2, 10)
        attention_mask[0, 5:] = 0  # Mask some positions

        loss = loss_fn(logits, labels, attention_mask)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_forward_all_labels_ignored(self):
        """Test forward returns zero when all labels are -100."""
        loss_fn = AudioTaskLoss(task_type="qa")
        logits = torch.randn(2, 10, 100)
        labels = torch.full((2, 10), -100)  # All ignored

        loss = loss_fn(logits, labels)

        assert loss.item() == 0.0
        assert loss.requires_grad  # Should still have requires_grad

    def test_forward_partial_labels_ignored(self):
        """Test forward handles partially ignored labels."""
        loss_fn = AudioTaskLoss(task_type="qa")
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        labels[:, :5] = -100  # Ignore first 5 positions

        loss = loss_fn(logits, labels)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_forward_out_of_vocab_labels_filtered(self):
        """Test forward filters labels outside vocab range."""
        loss_fn = AudioTaskLoss(task_type="qa")
        logits = torch.randn(2, 10, 100)  # vocab_size = 100
        labels = torch.randint(0, 100, (2, 10))
        labels[0, 3] = 150  # Out of vocab

        loss = loss_fn(logits, labels)

        assert not torch.isnan(loss)

    def test_forward_gradient_flow(self):
        """Test gradients flow through the loss."""
        loss_fn = AudioTaskLoss(task_type="qa")
        logits = torch.randn(2, 10, 100, requires_grad=True)
        labels = torch.randint(0, 100, (2, 10))

        loss = loss_fn(logits, labels)
        loss.backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_forward_shifted_labels(self):
        """Test that loss uses shifted logits and labels (causal LM style)."""
        loss_fn = AudioTaskLoss(task_type="qa")
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))

        # The loss should use logits[:-1] to predict labels[1:]
        # This test verifies the function runs without error
        loss = loss_fn(logits, labels)
        assert loss >= 0


class TestCombinedStageLoss:
    """Tests for CombinedStageLoss class."""

    @pytest.fixture
    def combined_loss(self):
        """Create a CombinedStageLoss instance."""
        retention_loss = RetentionLoss()
        audio_task_loss = AudioTaskLoss()
        return CombinedStageLoss(retention_loss, audio_task_loss)

    def test_initialization(self, combined_loss):
        """Test CombinedStageLoss initializes correctly."""
        assert combined_loss.audio_weight == 1.0
        assert combined_loss.retention_weight == 1.0
        assert combined_loss.debug is False

    def test_initialization_custom_weights(self):
        """Test CombinedStageLoss with custom weights."""
        retention_loss = RetentionLoss(distillation_weight=0.0)
        audio_task_loss = AudioTaskLoss()
        combined = CombinedStageLoss(
            retention_loss,
            audio_task_loss,
            audio_weight=0.5,
            retention_weight=0.0,
        )
        assert combined.audio_weight == 0.5
        assert combined.retention_weight == 0.0
        assert combined.retention_enabled is False

    def test_set_debug(self, combined_loss):
        """Test debug mode propagates to sub-losses."""
        combined_loss.set_debug(True)
        assert combined_loss.debug is True
        assert combined_loss.audio_task_loss.debug is True

    def test_forward_audio_only_batch(self, combined_loss):
        """Test forward with audio-only samples."""
        safe_outputs = {"logits": torch.randn(2, 10, 100)}
        base_outputs = {"logits": torch.randn(2, 10, 100)}
        batch = {
            "labels": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }
        has_audio = torch.tensor([True, True])

        result = combined_loss(safe_outputs, base_outputs, batch, has_audio)

        assert "total_loss" in result
        assert "audio_task_loss" in result
        assert "retention_loss" in result
        assert not torch.isnan(result["total_loss"])

    def test_forward_vl_only_batch(self, combined_loss):
        """Test forward with VL-only samples (no audio)."""
        safe_outputs = {"logits": torch.randn(2, 10, 100)}
        base_outputs = {"logits": torch.randn(2, 10, 100)}
        batch = {
            "labels": torch.randint(0, 100, (2, 10)),
            "attention_mask": torch.ones(2, 10),
        }
        has_audio = torch.tensor([False, False])

        result = combined_loss(safe_outputs, base_outputs, batch, has_audio)

        assert result["audio_task_loss"].item() == 0.0
        assert not torch.isnan(result["total_loss"])

    def test_forward_mixed_batch(self, combined_loss):
        """Test forward with mixed audio/VL batch."""
        safe_outputs = {"logits": torch.randn(4, 10, 100)}
        base_outputs = {"logits": torch.randn(4, 10, 100)}
        batch = {
            "labels": torch.randint(0, 100, (4, 10)),
            "attention_mask": torch.ones(4, 10),
        }
        has_audio = torch.tensor([True, False, True, False])

        result = combined_loss(safe_outputs, base_outputs, batch, has_audio)

        assert not torch.isnan(result["total_loss"])
        assert result["audio_task_loss"] > 0  # Some audio samples

    def test_forward_no_base_outputs(self):
        """Test forward without base outputs (retention disabled)."""
        retention_loss = RetentionLoss(distillation_weight=0.0)
        audio_task_loss = AudioTaskLoss()
        combined = CombinedStageLoss(
            retention_loss, audio_task_loss, retention_weight=0.0
        )

        safe_outputs = {"logits": torch.randn(2, 10, 100)}
        batch = {
            "labels": torch.randint(0, 100, (2, 10)),
        }
        has_audio = torch.tensor([True, True])

        result = combined(safe_outputs, None, batch, has_audio)

        assert result["retention_loss"].item() == 0.0

    def test_forward_raises_without_logits(self, combined_loss):
        """Test forward raises when SAFE outputs lack logits."""
        safe_outputs = {"hidden_states": torch.randn(2, 10, 100)}  # No logits
        base_outputs = {"logits": torch.randn(2, 10, 100)}
        batch = {"labels": torch.randint(0, 100, (2, 10))}
        has_audio = torch.tensor([True, True])

        with pytest.raises((ValueError, KeyError, TypeError)):
            combined_loss(safe_outputs, base_outputs, batch, has_audio)


class TestRewardFunction:
    """Tests for RewardFunction class."""

    def test_initialization(self):
        """Test RewardFunction initializes with default parameters."""
        reward_fn = RewardFunction()
        assert reward_fn.alpha == 0.3
        assert reward_fn.gamma == 0.5
        assert reward_fn.token_cost == 0.01
        assert reward_fn.latency_cost == 0.001

    def test_initialization_custom_params(self):
        """Test RewardFunction with custom parameters."""
        reward_fn = RewardFunction(alpha=0.5, gamma=0.3, token_cost=0.02)
        assert reward_fn.alpha == 0.5
        assert reward_fn.gamma == 0.3
        assert reward_fn.token_cost == 0.02

    def test_compute_score_qa_task(self):
        """Test score computation for QA task."""
        reward_fn = RewardFunction()
        predictions = torch.tensor([1, 2, 3, 1])
        targets = torch.tensor([1, 2, 0, 1])  # 3/4 correct

        scores = reward_fn.compute_score(predictions, targets, task_type="qa")

        assert scores.shape == (4,)
        assert scores.sum().item() == 3.0  # 3 correct

    def test_compute_score_qa_task_from_logits(self):
        """Test score computation from logits for QA task."""
        reward_fn = RewardFunction()
        # Create logits where argmax gives [0, 1, 2, 3]
        predictions = torch.zeros(4, 5)
        predictions[0, 0] = 10.0
        predictions[1, 1] = 10.0
        predictions[2, 2] = 10.0
        predictions[3, 3] = 10.0
        targets = torch.tensor([0, 1, 2, 0])  # 3/4 correct

        scores = reward_fn.compute_score(predictions, targets, task_type="qa")

        assert scores.sum().item() == 3.0

    def test_compute_score_caption_task(self):
        """Test score computation for caption task."""
        reward_fn = RewardFunction()
        predictions = torch.randint(0, 100, (2, 10, 100))
        targets = torch.randint(0, 100, (2, 10))

        scores = reward_fn.compute_score(predictions, targets, task_type="caption")

        assert scores.shape == (2,)
        assert (scores >= 0).all()
        assert (scores <= 1).all()

    def test_compute_score_invalid_task(self):
        """Test compute_score raises for invalid task type."""
        reward_fn = RewardFunction()
        with pytest.raises(ValueError, match="Unsupported task type"):
            reward_fn.compute_score(torch.tensor([1]), torch.tensor([1]), "invalid")

    def test_compute_latency_cost_tokens_only(self):
        """Test latency cost with token count only."""
        reward_fn = RewardFunction(token_cost=0.1)
        num_tokens = torch.tensor([0, 4, 8, 12])

        costs = reward_fn.compute_latency_cost(num_tokens)

        expected = torch.tensor([0.0, 0.4, 0.8, 1.2])
        assert torch.allclose(costs, expected)

    def test_compute_latency_cost_with_time(self):
        """Test latency cost with token count and processing time."""
        reward_fn = RewardFunction(token_cost=0.01, latency_cost=0.001)
        num_tokens = torch.tensor([8.0, 8.0])
        processing_time = torch.tensor([100.0, 200.0])  # ms

        costs = reward_fn.compute_latency_cost(num_tokens, processing_time)

        # 8 * 0.01 + time * 0.001
        expected = torch.tensor([0.08 + 0.1, 0.08 + 0.2])
        assert torch.allclose(costs, expected)

    def test_compute_irrelevance_penalty(self):
        """Test irrelevance penalty computation."""
        reward_fn = RewardFunction()
        used_audio = torch.tensor([True, True, False, False])
        is_audio_irrelevant = torch.tensor([True, False, True, False])

        penalties = reward_fn.compute_irrelevance_penalty(used_audio, is_audio_irrelevant)

        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(penalties, expected)

    def test_call_computes_total_reward(self):
        """Test __call__ computes total reward correctly."""
        reward_fn = RewardFunction(alpha=0.3, gamma=0.5)
        predictions = torch.tensor([1, 2])
        targets = torch.tensor([1, 2])  # All correct
        num_tokens = torch.tensor([4, 8])
        used_audio = torch.tensor([True, True])
        is_audio_irrelevant = torch.tensor([False, True])

        rewards = reward_fn(
            predictions, targets, num_tokens, used_audio, is_audio_irrelevant
        )

        assert rewards.shape == (2,)
        # Sample 1: 1.0 - 0.3 * 4 * 0.01 - 0.5 * 0 = 0.988
        # Sample 2: 1.0 - 0.3 * 8 * 0.01 - 0.5 * 1 = 0.476
        assert rewards[0] > rewards[1]  # Sample 2 has irrelevance penalty


class TestConstrainedRetentionLoss:
    """Tests for ConstrainedRetentionLoss class."""

    def test_initialization(self):
        """Test ConstrainedRetentionLoss initializes correctly."""
        loss_fn = ConstrainedRetentionLoss(baseline_score=0.9)
        assert loss_fn.baseline_score == 0.9
        assert loss_fn.tolerance == 0.003
        assert loss_fn.threshold == 0.9 - 0.003
        assert loss_fn.lambda_multiplier.item() == 0.0

    def test_initialization_custom_params(self):
        """Test ConstrainedRetentionLoss with custom parameters."""
        loss_fn = ConstrainedRetentionLoss(
            baseline_score=0.85,
            tolerance=0.01,
            lambda_lr=0.05,
            lambda_max=5.0,
        )
        assert loss_fn.baseline_score == 0.85
        assert loss_fn.tolerance == 0.01
        assert loss_fn.threshold == 0.84

    def test_update_vl_estimate(self):
        """Test VL estimate updates with EMA."""
        loss_fn = ConstrainedRetentionLoss(baseline_score=0.9)
        initial_ema = loss_fn.vl_score_ema

        loss_fn.update_vl_estimate(0.8)

        # EMA should move toward 0.8
        assert loss_fn.vl_score_ema < initial_ema
        assert loss_fn.vl_score_ema > 0.8  # But not all the way

    def test_compute_constraint_penalty_no_violation(self):
        """Test constraint penalty is zero when constraint satisfied."""
        loss_fn = ConstrainedRetentionLoss(baseline_score=0.9, tolerance=0.01)
        loss_fn.vl_score_ema = 0.9  # Above threshold

        penalty = loss_fn.compute_constraint_penalty()

        assert penalty.item() == 0.0

    def test_compute_constraint_penalty_with_violation(self):
        """Test constraint penalty is non-zero when constraint violated."""
        loss_fn = ConstrainedRetentionLoss(baseline_score=0.9, tolerance=0.01)
        loss_fn.vl_score_ema = 0.8  # Below threshold
        loss_fn.lambda_multiplier = torch.tensor(1.0)

        penalty = loss_fn.compute_constraint_penalty()

        assert penalty.item() > 0.0

    def test_update_lambda_increases_on_violation(self):
        """Test lambda increases when constraint is violated."""
        loss_fn = ConstrainedRetentionLoss(baseline_score=0.9, lambda_lr=0.1)
        loss_fn.vl_score_ema = 0.8  # Below threshold
        initial_lambda = float(loss_fn.lambda_multiplier)

        loss_fn.update_lambda()

        assert loss_fn.lambda_multiplier > initial_lambda

    def test_update_lambda_decreases_when_satisfied(self):
        """Test lambda decreases when constraint is satisfied."""
        loss_fn = ConstrainedRetentionLoss(baseline_score=0.9, lambda_lr=0.1)
        loss_fn.lambda_multiplier = torch.tensor(1.0)
        loss_fn.vl_score_ema = 0.95  # Above threshold

        loss_fn.update_lambda()

        assert loss_fn.lambda_multiplier < 1.0

    def test_update_lambda_clipped_to_max(self):
        """Test lambda is clipped to maximum value."""
        loss_fn = ConstrainedRetentionLoss(
            baseline_score=0.9, lambda_lr=100.0, lambda_max=5.0
        )
        loss_fn.vl_score_ema = 0.5  # Large violation

        loss_fn.update_lambda()

        assert loss_fn.lambda_multiplier <= 5.0

    def test_call_applies_penalty(self):
        """Test __call__ applies penalty to batch returns."""
        loss_fn = ConstrainedRetentionLoss(baseline_score=0.9)
        loss_fn.vl_score_ema = 0.8
        loss_fn.lambda_multiplier = torch.tensor(1.0)
        batch_returns = torch.tensor([1.0, 2.0, 3.0])

        constrained_returns = loss_fn(batch_returns)

        # Returns should be reduced by penalty
        assert (constrained_returns < batch_returns).all()
