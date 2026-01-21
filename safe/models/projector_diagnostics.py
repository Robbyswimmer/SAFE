"""
Projector Ablation Diagnostics for SAFE Audio-LLM.

Diagnoses WHERE and HOW discriminative information is lost
when CLAP embeddings are projected to LLM token space.

Key diagnostics:
1. Pairwise distance preservation - does projector preserve geometry?
2. Effective rank (SVD) - is output collapsing to low-dim subspace?
3. Token diversity - are all tokens identical?
4. Sensitivity / Jacobian - is projector saturating?
5. Dimension variance - how many dimensions are dead?
6. Audio influence - does audio change LLM logits?
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProjectorDiagnostics:
    """Container for all diagnostic metrics."""

    # Pairwise distance preservation
    distance_correlation: float = 0.0  # corr(sim_encoder, sim_projector)
    encoder_sim_std: float = 0.0       # std of encoder pairwise similarities
    projector_sim_std: float = 0.0     # std of projector pairwise similarities (collapse = low)

    # Effective rank / singular values
    effective_rank: float = 0.0        # exp(entropy of normalized singular values)
    top_singular_values: List[float] = field(default_factory=list)  # Top-k singular values
    singular_value_entropy: float = 0.0     # Entropy of singular value distribution

    # Token diversity
    mean_intra_sample_similarity: float = 0.0  # Mean pairwise cosine among tokens within sample
    per_token_variance: List[float] = field(default_factory=list)  # Variance of each token position across batch
    dead_token_count: int = 0                # Tokens with near-zero variance

    # Sensitivity / Jacobian
    input_output_sensitivity: float = 0.0  # ||delta_output|| / ||delta_input||
    gradient_norm: float = 0.0             # Mean gradient norm through projector
    saturation_ratio: float = 0.0          # Fraction of near-zero gradients

    # Variance per dimension
    active_dimension_count: int = 0      # Dims with variance > threshold
    dead_dimension_count: int = 0        # Dims with near-zero variance
    dimension_variance_entropy: float = 0.0  # Entropy of variance distribution

    # Audio influence on logits (requires model forward pass)
    logits_delta_norm: Optional[float] = None  # ||logits_on - logits_off||
    logits_cosine_similarity: Optional[float] = None


class ProjectorDiagnosticsComputer:
    """Computes comprehensive diagnostics for the audio projector."""

    def __init__(
        self,
        variance_threshold: float = 1e-6,
        sensitivity_perturbation_scale: float = 0.01,
        top_k_singular_values: int = 10,
    ):
        self.variance_threshold = variance_threshold
        self.perturbation_scale = sensitivity_perturbation_scale
        self.top_k = top_k_singular_values

    @torch.no_grad()
    def compute_all(
        self,
        encoder_outputs: torch.Tensor,  # (N, encoder_dim) e.g. (128, 512)
        projector_outputs: torch.Tensor,  # (N, num_tokens, hidden_dim) e.g. (128, 16, 5120)
        projector: Optional[nn.Module] = None,
    ) -> ProjectorDiagnostics:
        """Compute all diagnostics in a single pass."""

        # Ensure float32 for numerical stability
        encoder_outputs = encoder_outputs.float()
        projector_outputs = projector_outputs.float()

        # 1. Pairwise distance preservation
        dist_corr, enc_std, proj_std = self._compute_distance_preservation(
            encoder_outputs, projector_outputs
        )

        # 2. Effective rank / SVD analysis
        eff_rank, top_svs, sv_entropy = self._compute_effective_rank(projector_outputs)

        # 3. Token diversity
        intra_sim, per_token_var, dead_tokens = self._compute_token_diversity(
            projector_outputs
        )

        # 4. Sensitivity / Jacobian (requires projector module)
        if projector is not None:
            sensitivity, grad_norm, saturation = self._compute_sensitivity(
                encoder_outputs, projector
            )
        else:
            sensitivity, grad_norm, saturation = 0.0, 0.0, 0.0

        # 5. Variance per dimension
        active_dims, dead_dims, var_entropy = self._compute_dimension_variance(
            projector_outputs
        )

        return ProjectorDiagnostics(
            distance_correlation=dist_corr,
            encoder_sim_std=enc_std,
            projector_sim_std=proj_std,
            effective_rank=eff_rank,
            top_singular_values=top_svs,
            singular_value_entropy=sv_entropy,
            mean_intra_sample_similarity=intra_sim,
            per_token_variance=per_token_var,
            dead_token_count=dead_tokens,
            input_output_sensitivity=sensitivity,
            gradient_norm=grad_norm,
            saturation_ratio=saturation,
            active_dimension_count=active_dims,
            dead_dimension_count=dead_dims,
            dimension_variance_entropy=var_entropy,
        )

    def _compute_distance_preservation(
        self,
        encoder_outputs: torch.Tensor,
        projector_outputs: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """
        Compute correlation between pairwise similarities in encoder vs projector space.

        If this is low, the projector is destroying the geometry that makes
        different audio samples distinguishable.
        """
        N = encoder_outputs.size(0)
        if N < 2:
            return 0.0, 0.0, 0.0

        # Encoder pairwise cosine similarities
        enc_norm = F.normalize(encoder_outputs, dim=-1)
        enc_sim = torch.mm(enc_norm, enc_norm.t())  # (N, N)

        # Projector: flatten tokens then compute similarities
        proj_flat = projector_outputs.view(N, -1)  # (N, num_tokens * hidden_dim)
        proj_norm = F.normalize(proj_flat, dim=-1)
        proj_sim = torch.mm(proj_norm, proj_norm.t())  # (N, N)

        # Extract upper triangle (excluding diagonal)
        triu_idx = torch.triu_indices(N, N, offset=1, device=encoder_outputs.device)
        enc_upper = enc_sim[triu_idx[0], triu_idx[1]]
        proj_upper = proj_sim[triu_idx[0], triu_idx[1]]

        # Pearson correlation
        enc_mean = enc_upper.mean()
        proj_mean = proj_upper.mean()
        enc_centered = enc_upper - enc_mean
        proj_centered = proj_upper - proj_mean

        numerator = (enc_centered * proj_centered).sum()
        denominator = (enc_centered.pow(2).sum().sqrt() *
                       proj_centered.pow(2).sum().sqrt() + 1e-8)
        correlation = (numerator / denominator).item()

        return correlation, enc_upper.std().item(), proj_upper.std().item()

    def _compute_effective_rank(
        self,
        projector_outputs: torch.Tensor,
    ) -> Tuple[float, List[float], float]:
        """
        Compute effective rank via SVD entropy.

        Low effective rank = representations are collapsing to a low-dimensional subspace.
        """
        N = projector_outputs.size(0)
        proj_flat = projector_outputs.view(N, -1).float()  # (N, D)

        # Center the data
        proj_centered = proj_flat - proj_flat.mean(dim=0, keepdim=True)

        # SVD (use economic SVD for efficiency)
        try:
            # Limit matrix size for memory
            max_dim = min(proj_centered.shape[0], proj_centered.shape[1], 1000)
            if proj_centered.shape[1] > max_dim:
                # Random projection for large matrices
                proj_matrix = torch.randn(proj_centered.shape[1], max_dim, device=proj_centered.device)
                proj_matrix = proj_matrix / proj_matrix.norm(dim=0, keepdim=True)
                proj_reduced = proj_centered @ proj_matrix
            else:
                proj_reduced = proj_centered

            U, S, Vh = torch.linalg.svd(proj_reduced, full_matrices=False)
        except RuntimeError:
            # Fallback if SVD fails
            return 1.0, [0.0] * self.top_k, 0.0

        # Normalize singular values to form probability distribution
        S_sum = S.sum() + 1e-8
        S_normalized = S / S_sum

        # Entropy of normalized singular values
        entropy = -(S_normalized * (S_normalized + 1e-10).log()).sum().item()

        # Effective rank = exp(entropy)
        effective_rank = math.exp(entropy)

        # Top-k singular values
        top_svs = S[:self.top_k].tolist()

        return effective_rank, top_svs, entropy

    def _compute_token_diversity(
        self,
        projector_outputs: torch.Tensor,
    ) -> Tuple[float, List[float], int]:
        """
        Analyze token diversity:
        1. Intra-sample similarity: Are tokens within the same sample too similar?
        2. Per-token variance: Are some token positions "dead" (constant across samples)?
        """
        B, T, D = projector_outputs.shape

        # 1. Mean pairwise cosine among tokens within same sample
        intra_sims = []
        for i in range(min(B, 32)):  # Limit for speed
            tokens = projector_outputs[i]  # (T, D)
            tokens_norm = F.normalize(tokens, dim=-1)
            sim_matrix = torch.mm(tokens_norm, tokens_norm.t())  # (T, T)
            # Upper triangle excluding diagonal
            if T > 1:
                triu_idx = torch.triu_indices(T, T, offset=1, device=tokens.device)
                pairwise = sim_matrix[triu_idx[0], triu_idx[1]]
                intra_sims.append(pairwise.mean().item())
        mean_intra_sim = sum(intra_sims) / len(intra_sims) if intra_sims else 0.0

        # 2. Per-token variance across batch
        per_token_var = []
        dead_count = 0
        for t in range(T):
            token_t = projector_outputs[:, t, :]  # (B, D)
            var = token_t.var(dim=0).mean().item()  # Mean variance across dimensions
            per_token_var.append(var)
            if var < self.variance_threshold:
                dead_count += 1

        return mean_intra_sim, per_token_var, dead_count

    def _compute_sensitivity(
        self,
        encoder_outputs: torch.Tensor,
        projector: nn.Module,
    ) -> Tuple[float, float, float]:
        """
        Compute input/output sensitivity via perturbation analysis.

        This detects if the projector is saturating (small input changes
        produce zero/tiny output changes).
        """
        # Clone and enable gradients for this computation
        encoder_clone = encoder_outputs.detach().clone().requires_grad_(True)

        try:
            # Forward pass
            original_output = projector(encoder_clone.float())

            # Create perturbation
            perturbation = torch.randn_like(encoder_clone) * self.perturbation_scale
            perturbed_input = encoder_clone + perturbation

            with torch.no_grad():
                perturbed_output = projector(perturbed_input.float())

            # Compute relative output change
            input_delta = perturbation.norm().item()
            output_delta = (perturbed_output - original_output.detach()).norm().item()
            sensitivity = output_delta / (input_delta + 1e-8)

            # Compute gradient norm
            loss = original_output.sum()
            loss.backward()
            grad_norm = encoder_clone.grad.norm().item() if encoder_clone.grad is not None else 0.0

            # Saturation: fraction of near-zero gradients
            if encoder_clone.grad is not None:
                saturation = (encoder_clone.grad.abs() < 1e-6).float().mean().item()
            else:
                saturation = 1.0

            return sensitivity, grad_norm, saturation

        except Exception as e:
            print(f"[Diagnostics] Sensitivity computation failed: {e}")
            return 0.0, 0.0, 1.0

    def _compute_dimension_variance(
        self,
        projector_outputs: torch.Tensor,
    ) -> Tuple[int, int, float]:
        """
        Analyze variance per output dimension.

        Dead dimensions = dimensions that are constant across all samples.
        These represent wasted capacity.
        """
        B, T, D = projector_outputs.shape

        # Flatten to (B, T*D) and compute variance per dimension
        flat = projector_outputs.view(B, -1)  # (B, T*D)
        var_per_dim = flat.var(dim=0)  # (T*D,)

        active = (var_per_dim > self.variance_threshold).sum().item()
        dead = (var_per_dim <= self.variance_threshold).sum().item()

        # Entropy of variance distribution
        var_sum = var_per_dim.sum() + 1e-8
        var_normalized = var_per_dim / var_sum
        entropy = -(var_normalized * (var_normalized + 1e-10).log()).sum().item()

        return int(active), int(dead), entropy


def format_diagnostics_report(diag: ProjectorDiagnostics) -> str:
    """Format diagnostics as a human-readable report."""

    # Determine health status for each metric
    dist_status = "HEALTHY" if diag.distance_correlation > 0.5 else "COLLAPSE: geometry destroyed"
    rank_status = "HEALTHY" if diag.effective_rank > 10 else "LOW RANK: representations collapsing"
    token_status = "HEALTHY" if diag.mean_intra_sample_similarity < 0.9 else "TOKEN COLLAPSE: tokens too similar"
    sat_status = "HEALTHY" if diag.saturation_ratio < 0.5 else "SATURATED: gradients vanishing"

    dead_ratio = diag.dead_dimension_count / max(diag.active_dimension_count + diag.dead_dimension_count, 1)
    dim_status = "HEALTHY" if dead_ratio < 0.1 else "MANY DEAD DIMS"

    lines = [
        "=" * 70,
        "PROJECTOR ABLATION DIAGNOSTICS",
        "=" * 70,
        "",
        "1. PAIRWISE DISTANCE PRESERVATION",
        f"   Correlation(encoder_sim, projector_sim): {diag.distance_correlation:.4f}",
        f"   Encoder similarity std:    {diag.encoder_sim_std:.4f}",
        f"   Projector similarity std:  {diag.projector_sim_std:.4f}",
        f"   --> {dist_status}",
        "",
        "2. EFFECTIVE RANK (SVD Analysis)",
        f"   Effective rank:            {diag.effective_rank:.2f}",
        f"   Singular value entropy:    {diag.singular_value_entropy:.4f}",
        f"   Top singular values:       {[f'{v:.2f}' for v in diag.top_singular_values[:5]]}",
        f"   --> {rank_status}",
        "",
        "3. TOKEN DIVERSITY",
        f"   Mean intra-sample cosine:  {diag.mean_intra_sample_similarity:.4f}",
        f"   Dead tokens (no variance): {diag.dead_token_count}",
        f"   Per-token variance (first 5): {[f'{v:.6f}' for v in diag.per_token_variance[:5]]}",
        f"   --> {token_status}",
        "",
        "4. SENSITIVITY / JACOBIAN",
        f"   Input-output sensitivity:  {diag.input_output_sensitivity:.4f}",
        f"   Gradient norm:             {diag.gradient_norm:.4f}",
        f"   Saturation ratio:          {diag.saturation_ratio:.4f}",
        f"   --> {sat_status}",
        "",
        "5. DIMENSION VARIANCE",
        f"   Active dimensions:         {diag.active_dimension_count}",
        f"   Dead dimensions:           {diag.dead_dimension_count}",
        f"   Dead ratio:                {dead_ratio:.2%}",
        f"   Variance entropy:          {diag.dimension_variance_entropy:.4f}",
        f"   --> {dim_status}",
        "",
    ]

    if diag.logits_delta_norm is not None:
        audio_status = "AUDIO MATTERS" if diag.logits_delta_norm > 0.1 else "AUDIO IGNORED"
        lines.extend([
            "6. AUDIO INFLUENCE ON LOGITS",
            f"   ||logits_on - logits_off||: {diag.logits_delta_norm:.4f}",
            f"   Cosine similarity:          {diag.logits_cosine_similarity:.4f}",
            f"   --> {audio_status}",
            "",
        ])

    lines.append("=" * 70)
    return "\n".join(lines)


def diagnostics_to_wandb_dict(
    diag: ProjectorDiagnostics,
    prefix: str = "projector_ablation/",
) -> Dict[str, float]:
    """Convert diagnostics to wandb-loggable dictionary."""
    result = {
        f"{prefix}distance_correlation": diag.distance_correlation,
        f"{prefix}encoder_sim_std": diag.encoder_sim_std,
        f"{prefix}projector_sim_std": diag.projector_sim_std,
        f"{prefix}effective_rank": diag.effective_rank,
        f"{prefix}singular_value_entropy": diag.singular_value_entropy,
        f"{prefix}mean_intra_sample_similarity": diag.mean_intra_sample_similarity,
        f"{prefix}dead_token_count": float(diag.dead_token_count),
        f"{prefix}input_output_sensitivity": diag.input_output_sensitivity,
        f"{prefix}gradient_norm": diag.gradient_norm,
        f"{prefix}saturation_ratio": diag.saturation_ratio,
        f"{prefix}active_dimension_count": float(diag.active_dimension_count),
        f"{prefix}dead_dimension_count": float(diag.dead_dimension_count),
        f"{prefix}dimension_variance_entropy": diag.dimension_variance_entropy,
    }

    # Add top singular values
    for i, sv in enumerate(diag.top_singular_values[:5]):
        result[f"{prefix}singular_value_{i}"] = sv

    # Add per-token variance (first 8)
    for i, var in enumerate(diag.per_token_variance[:8]):
        result[f"{prefix}token_{i}_variance"] = var

    # Add logits delta if available
    if diag.logits_delta_norm is not None:
        result[f"{prefix}logits_delta_norm"] = diag.logits_delta_norm
    if diag.logits_cosine_similarity is not None:
        result[f"{prefix}logits_cosine_similarity"] = diag.logits_cosine_similarity

    return result


def get_diagnostic_summary(diag: ProjectorDiagnostics) -> Dict[str, str]:
    """Get a summary of diagnosed issues."""
    issues = []

    if diag.distance_correlation < 0.3:
        issues.append("CRITICAL: Distance correlation < 0.3 - projector destroys geometry")
    elif diag.distance_correlation < 0.5:
        issues.append("WARNING: Distance correlation < 0.5 - geometry weakly preserved")

    if diag.effective_rank < 5:
        issues.append("CRITICAL: Effective rank < 5 - severe dimensional collapse")
    elif diag.effective_rank < 10:
        issues.append("WARNING: Effective rank < 10 - moderate collapse")

    if diag.mean_intra_sample_similarity > 0.95:
        issues.append("CRITICAL: Intra-sample similarity > 0.95 - tokens are identical")
    elif diag.mean_intra_sample_similarity > 0.9:
        issues.append("WARNING: Intra-sample similarity > 0.9 - tokens very similar")

    if diag.saturation_ratio > 0.8:
        issues.append("WARNING: Saturation ratio > 0.8 - gradients vanishing")

    if diag.projector_sim_std < 0.01:
        issues.append("WARNING: Projector similarity std < 0.01 - all outputs look same")

    dead_ratio = diag.dead_dimension_count / max(diag.active_dimension_count + diag.dead_dimension_count, 1)
    if dead_ratio > 0.5:
        issues.append(f"WARNING: {dead_ratio:.0%} of dimensions are dead")

    if diag.logits_delta_norm is not None and diag.logits_delta_norm < 0.01:
        issues.append("CRITICAL: Audio has near-zero influence on logits")

    return {
        "issues": issues,
        "critical_count": sum(1 for i in issues if i.startswith("CRITICAL")),
        "warning_count": sum(1 for i in issues if i.startswith("WARNING")),
        "healthy": len(issues) == 0,
    }
