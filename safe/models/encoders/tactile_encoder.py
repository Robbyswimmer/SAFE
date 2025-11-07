from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class TactileEncoderOutput:
    """Container for tactile encoder results."""

    embeddings: torch.Tensor
    sequence_embeddings: Optional[torch.Tensor]


class TactileEncoder(nn.Module):
    """Encode tactile time-series using a lightweight GRU backbone."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        embed_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_size <= 0:
            raise ValueError("input_size must be positive")

        self.bidirectional = bidirectional
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        output_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(output_dim, embed_dim)

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        *,
        return_sequence: bool = False,
    ) -> TactileEncoderOutput:
        if sequence.ndim != 3:
            raise ValueError("Tactile sequence must have shape (batch, timesteps, features)")

        sequence = sequence.float()
        batch_size = sequence.size(0)

        if lengths is not None:
            lengths = lengths.to(sequence.device)
            packed = pack_padded_sequence(
                sequence,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, packed_hidden = self.rnn(packed)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            hidden = packed_hidden
        else:
            output, hidden = self.rnn(sequence)

        if self.bidirectional:
            # Concatenate the last forward and backward hidden states
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            last_hidden = hidden[-1]

        embeddings = self.proj(last_hidden)
        if embeddings.size(0) != batch_size:
            embeddings = embeddings.view(batch_size, -1)

        sequence_embeddings = output if return_sequence else None
        return TactileEncoderOutput(embeddings=embeddings, sequence_embeddings=sequence_embeddings)
