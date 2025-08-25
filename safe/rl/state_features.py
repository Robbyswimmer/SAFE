import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import re


class TextQuestionClassifier(nn.Module):
    """
    Learned classifier to categorize question types for RL controller.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.question_types = [
            "audio_dependent",   # Requires audio to answer
            "visual_only",       # Visual information sufficient
            "text_only",         # Can be answered from text
            "multimodal",        # Requires both audio and visual
            "ambiguous"          # Unclear modality requirements
        ]
        
        # Simple embedding-based classifier
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Simple keyword-based features as backup
        self.audio_keywords = [
            "sound", "hear", "audio", "noise", "music", "voice", "speak", 
            "sing", "play", "instrument", "loud", "quiet", "volume"
        ]
        self.visual_keywords = [
            "see", "look", "color", "shape", "size", "appear", "show",
            "visible", "bright", "dark", "red", "blue", "green"
        ]
        
    def _extract_keyword_features(self, questions: List[str]) -> torch.Tensor:
        """Extract keyword-based features from questions."""
        features = []
        
        for question in questions:
            question_lower = question.lower()
            
            # Count audio-related keywords
            audio_count = sum(1 for word in self.audio_keywords if word in question_lower)
            
            # Count visual-related keywords  
            visual_count = sum(1 for word in self.visual_keywords if word in question_lower)
            
            # Simple heuristics
            question_len = len(question.split())
            has_wh_word = any(q in question_lower for q in ["what", "how", "where", "when", "why", "which"])
            
            features.append([
                audio_count,
                visual_count, 
                question_len,
                int(has_wh_word),
                audio_count / max(question_len, 1),  # Audio density
                visual_count / max(question_len, 1)  # Visual density
            ])
            
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self, question_ids: torch.Tensor, questions: List[str] = None) -> torch.Tensor:
        """
        Classify question types.
        
        Args:
            question_ids: Tokenized question IDs (batch_size, seq_len)
            questions: Raw question strings for keyword features
            
        Returns:
            logits: (batch_size, num_classes) classification logits
        """
        # LSTM-based classification
        embedded = self.embedding(question_ids)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use final hidden state
        lstm_features = hidden[-1]  # (batch_size, hidden_dim)
        
        # Get classification logits
        logits = self.classifier(lstm_features)
        
        # Optionally incorporate keyword features
        if questions is not None:
            keyword_features = self._extract_keyword_features(questions)
            keyword_features = keyword_features.to(logits.device)
            
            # Simple feature combination (in practice, you might train this)
            audio_bias = keyword_features[:, 0] * 0.1  # Audio keyword boost
            logits[:, 0] += audio_bias  # Boost audio_dependent class
            
        return logits
    
    def predict_question_type(self, questions: List[str], question_ids: torch.Tensor = None) -> List[str]:
        """
        Predict question types from raw text.
        
        Args:
            questions: List of question strings
            question_ids: Optional tokenized questions
            
        Returns:
            List of predicted question types
        """
        if question_ids is not None:
            with torch.no_grad():
                logits = self.forward(question_ids, questions)
                predictions = torch.argmax(logits, dim=-1)
        else:
            # Fallback to keyword-based classification
            predictions = []
            for question in questions:
                keyword_features = self._extract_keyword_features([question])
                audio_score = keyword_features[0, 0] + keyword_features[0, 4] * 10
                visual_score = keyword_features[0, 1] + keyword_features[0, 5] * 10
                
                if audio_score > visual_score and audio_score > 1:
                    predictions.append(0)  # audio_dependent
                elif visual_score > 1:
                    predictions.append(1)  # visual_only
                else:
                    predictions.append(2)  # text_only
                    
            predictions = torch.tensor(predictions)
        
        return [self.question_types[pred.item()] for pred in predictions]


class StateFeatureExtractor(nn.Module):
    """
    Extracts state features for the RL controller policy.
    
    Features include:
    - Text question type (learned classifier)
    - CLIP image embedding 
    - Base VL logits/confidence (entropy/top-2 margin)
    - CLAP audio preview (64-dim PCA)
    - Simple audio stats (energy/voicing)
    - Visual-text alignment score
    """
    
    def __init__(
        self,
        question_classifier: TextQuestionClassifier,
        clip_embed_dim: int = 512,
        clap_embed_dim: int = 512,
        audio_stats_dim: int = 10,
        pca_components: int = 64,
        output_dim: int = 256
    ):
        super().__init__()
        
        self.question_classifier = question_classifier
        self.clip_embed_dim = clip_embed_dim
        self.clap_embed_dim = clap_embed_dim
        self.pca_components = pca_components
        self.audio_stats_dim = audio_stats_dim
        
        # PCA for CLAP dimensionality reduction
        self.clap_pca = PCA(n_components=pca_components)
        self.clap_pca_fitted = False
        
        # Feature dimensions
        self.question_type_dim = question_classifier.num_classes
        self.vl_confidence_dim = 3  # entropy, top-2 margin, max prob
        self.alignment_dim = 1
        
        total_feature_dim = (
            self.question_type_dim +
            clip_embed_dim +
            self.vl_confidence_dim +
            pca_components +
            audio_stats_dim +
            self.alignment_dim
        )
        
        # Feature fusion network
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_feature_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
    def _compute_vl_confidence_features(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence features from VL model logits.
        
        Args:
            logits: (batch_size, vocab_size) model logits
            
        Returns:
            confidence_features: (batch_size, 3) [entropy, top2_margin, max_prob]
        """
        probs = torch.softmax(logits, dim=-1)
        
        # Entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
        # Top-2 margin
        top2_probs, _ = torch.topk(probs, k=2, dim=-1)
        top2_margin = top2_probs[:, 0] - top2_probs[:, 1]
        
        # Max probability
        max_prob = torch.max(probs, dim=-1)[0]
        
        return torch.stack([entropy, top2_margin, max_prob], dim=-1)
    
    def _compute_audio_stats(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute simple audio statistics.
        
        Args:
            audio: (batch_size, audio_length) raw audio
            
        Returns:
            audio_stats: (batch_size, audio_stats_dim) statistics
        """
        batch_size = audio.shape[0]
        stats = []
        
        for i in range(batch_size):
            audio_sample = audio[i].cpu().numpy()
            
            # Energy-based features
            energy = np.mean(audio_sample ** 2)
            rms_energy = np.sqrt(energy)
            
            # Zero-crossing rate (simple voicing indicator)
            zero_crossings = np.sum(np.diff(np.signbit(audio_sample)))
            zcr = zero_crossings / len(audio_sample)
            
            # Spectral features (simplified)
            fft = np.fft.fft(audio_sample)
            magnitude = np.abs(fft[:len(fft)//2])
            spectral_centroid = np.sum(magnitude * np.arange(len(magnitude))) / (np.sum(magnitude) + 1e-8)
            spectral_rolloff = np.percentile(magnitude, 85)
            
            # Basic statistics
            audio_mean = np.mean(audio_sample)
            audio_std = np.std(audio_sample)
            audio_max = np.max(np.abs(audio_sample))
            
            # Dynamic range
            dynamic_range = audio_max - np.min(np.abs(audio_sample))
            
            sample_stats = [
                energy, rms_energy, zcr, spectral_centroid, spectral_rolloff,
                audio_mean, audio_std, audio_max, dynamic_range, len(audio_sample)
            ]
            
            stats.append(sample_stats)
            
        return torch.tensor(stats, dtype=torch.float32, device=audio.device)
    
    def _compute_visual_text_alignment(
        self, 
        clip_features: torch.Tensor,
        question_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alignment score between visual and text features.
        
        Args:
            clip_features: (batch_size, clip_embed_dim) CLIP image features
            question_embeddings: (batch_size, embed_dim) question embeddings
            
        Returns:
            alignment_scores: (batch_size, 1) alignment scores
        """
        # Simple cosine similarity
        clip_norm = torch.nn.functional.normalize(clip_features, dim=-1)
        question_norm = torch.nn.functional.normalize(question_embeddings, dim=-1)
        
        # If dimensions don't match, project question embeddings
        if clip_features.shape[-1] != question_embeddings.shape[-1]:
            if not hasattr(self, 'question_projector'):
                self.question_projector = nn.Linear(
                    question_embeddings.shape[-1], 
                    clip_features.shape[-1]
                ).to(clip_features.device)
            question_norm = torch.nn.functional.normalize(
                self.question_projector(question_norm), dim=-1
            )
        
        alignment = torch.sum(clip_norm * question_norm, dim=-1, keepdim=True)
        
        return alignment
    
    def forward(
        self,
        questions: List[str],
        question_ids: torch.Tensor,
        question_embeddings: torch.Tensor,
        clip_features: torch.Tensor,
        vl_logits: torch.Tensor,
        clap_features: torch.Tensor,
        audio: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract state features for RL controller.
        
        Args:
            questions: List of question strings
            question_ids: Tokenized questions
            question_embeddings: Question embeddings from LLM
            clip_features: CLIP image features
            vl_logits: Base VL model logits
            clap_features: CLAP audio features
            audio: Raw audio waveform
            
        Returns:
            state_features: (batch_size, output_dim) state representation
        """
        batch_size = len(questions)
        device = clip_features.device
        
        # 1. Question type classification
        question_type_logits = self.question_classifier(question_ids, questions)
        question_type_probs = torch.softmax(question_type_logits, dim=-1)
        
        # 2. VL confidence features
        vl_confidence = self._compute_vl_confidence_features(vl_logits)
        
        # 3. CLAP audio preview (PCA)
        clap_np = clap_features.cpu().numpy()
        if not self.clap_pca_fitted and clap_np.shape[0] > 1:
            self.clap_pca.fit(clap_np)
            self.clap_pca_fitted = True
            
        if self.clap_pca_fitted:
            clap_pca = self.clap_pca.transform(clap_np)
        else:
            # Fallback: simple projection
            clap_pca = clap_np[:, :self.pca_components]
            if clap_pca.shape[1] < self.pca_components:
                padding = np.zeros((clap_pca.shape[0], self.pca_components - clap_pca.shape[1]))
                clap_pca = np.concatenate([clap_pca, padding], axis=1)
                
        clap_pca_tensor = torch.tensor(clap_pca, dtype=torch.float32, device=device)
        
        # 4. Audio statistics
        audio_stats = self._compute_audio_stats(audio)
        
        # 5. Visual-text alignment
        alignment_score = self._compute_visual_text_alignment(clip_features, question_embeddings)
        
        # Concatenate all features
        all_features = torch.cat([
            question_type_probs,      # (batch_size, question_type_dim)
            clip_features,            # (batch_size, clip_embed_dim)
            vl_confidence,            # (batch_size, 3)
            clap_pca_tensor,          # (batch_size, pca_components)
            audio_stats,              # (batch_size, audio_stats_dim)
            alignment_score           # (batch_size, 1)
        ], dim=-1)
        
        # Fuse features
        state_features = self.feature_fusion(all_features)
        
        return state_features
    
    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features for interpretability."""
        names = []
        
        # Question type features
        for qtype in self.question_classifier.question_types:
            names.append(f"question_type_{qtype}")
            
        # CLIP features
        for i in range(self.clip_embed_dim):
            names.append(f"clip_feature_{i}")
            
        # VL confidence features
        names.extend(["vl_entropy", "vl_top2_margin", "vl_max_prob"])
        
        # CLAP PCA features
        for i in range(self.pca_components):
            names.append(f"clap_pca_{i}")
            
        # Audio statistics
        audio_stat_names = [
            "audio_energy", "audio_rms", "zero_crossing_rate", 
            "spectral_centroid", "spectral_rolloff", "audio_mean",
            "audio_std", "audio_max", "dynamic_range", "audio_length"
        ]
        names.extend(audio_stat_names)
        
        # Alignment score
        names.append("visual_text_alignment")
        
        return names
    
    def fit_clap_pca(self, clap_features_list: List[torch.Tensor]):
        """
        Fit PCA on a collection of CLAP features.
        
        Args:
            clap_features_list: List of CLAP feature tensors
        """
        all_features = torch.cat(clap_features_list, dim=0).cpu().numpy()
        self.clap_pca.fit(all_features)
        self.clap_pca_fitted = True