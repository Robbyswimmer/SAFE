"""
Advanced Mock Data Generators for SAFE Testing

This module provides sophisticated mock datasets that simulate real-world
multimodal data with controllable difficulty levels and realistic characteristics.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import random
from collections import defaultdict

from safe.data.curriculum import DifficultyLevel


class QuestionType(Enum):
    """Types of questions for different modalities."""
    AUDIO_DEPENDENT = "audio_dependent"
    VISUAL_ONLY = "visual_only"
    AUDIO_VISUAL = "audio_visual"
    AUDIO_IRRELEVANT = "audio_irrelevant"
    TEXT_ONLY = "text_only"


@dataclass
class MockSampleSpec:
    """Specification for generating a mock sample."""
    question_type: QuestionType
    difficulty: DifficultyLevel
    has_audio: bool = True
    has_images: bool = True
    audio_duration: float = 10.0
    audio_complexity: str = "simple"  # simple, moderate, complex
    image_complexity: str = "simple"  # simple, moderate, complex
    question_complexity: str = "simple"  # simple, moderate, complex


class AudioGenerator:
    """Generates realistic mock audio data."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def generate_audio(self, duration: float, complexity: str = "simple", 
                      audio_type: str = "environmental") -> torch.Tensor:
        """Generate mock audio with specified characteristics.
        
        Args:
            duration: Audio duration in seconds
            complexity: Audio complexity level
            audio_type: Type of audio (environmental, speech, music)
            
        Returns:
            Audio tensor of shape (samples,)
        """
        num_samples = int(duration * self.sample_rate)
        
        if complexity == "simple":
            # Simple sine wave or noise
            if audio_type == "speech":
                audio = self._generate_simple_speech(num_samples)
            elif audio_type == "music":
                audio = self._generate_simple_music(num_samples)
            else:
                audio = self._generate_simple_environmental(num_samples)
                
        elif complexity == "moderate":
            # Mixed signals
            audio = self._generate_moderate_audio(num_samples, audio_type)
            
        else:  # complex
            # Complex multi-component audio
            audio = self._generate_complex_audio(num_samples, audio_type)
            
        return audio
        
    def _generate_simple_speech(self, num_samples: int) -> torch.Tensor:
        """Generate simple speech-like audio."""
        # Fundamental frequency around 150Hz for speech
        t = torch.linspace(0, num_samples / self.sample_rate, num_samples)
        
        # Basic formant structure
        f1, f2 = 800, 1200  # Typical formants
        speech = (0.3 * torch.sin(2 * np.pi * f1 * t) + 
                 0.2 * torch.sin(2 * np.pi * f2 * t))
        
        # Add envelope and noise
        envelope = torch.exp(-t * 0.5)  # Decay
        noise = 0.05 * torch.randn(num_samples)
        
        return speech * envelope + noise
        
    def _generate_simple_music(self, num_samples: int) -> torch.Tensor:
        """Generate simple music-like audio."""
        t = torch.linspace(0, num_samples / self.sample_rate, num_samples)
        
        # Musical notes (C major scale)
        frequencies = [261.63, 293.66, 329.63, 349.23, 392.00]  # C, D, E, F, G
        
        audio = torch.zeros(num_samples)
        for i, freq in enumerate(frequencies):
            start_idx = i * num_samples // len(frequencies)
            end_idx = (i + 1) * num_samples // len(frequencies)
            segment_t = t[start_idx:end_idx] - t[start_idx]
            
            # Add harmonic content
            note = (0.5 * torch.sin(2 * np.pi * freq * segment_t) +
                   0.3 * torch.sin(2 * np.pi * freq * 2 * segment_t) +
                   0.1 * torch.sin(2 * np.pi * freq * 3 * segment_t))
            
            audio[start_idx:end_idx] = note
            
        return audio
        
    def _generate_simple_environmental(self, num_samples: int) -> torch.Tensor:
        """Generate simple environmental audio."""
        # Wind/nature-like noise with spectral shaping
        noise = torch.randn(num_samples)
        
        # Simple low-pass filtering effect
        filtered = torch.zeros_like(noise)
        alpha = 0.1  # Filter coefficient
        
        for i in range(1, len(noise)):
            filtered[i] = alpha * noise[i] + (1 - alpha) * filtered[i-1]
            
        return 0.5 * filtered
        
    def _generate_moderate_audio(self, num_samples: int, audio_type: str) -> torch.Tensor:
        """Generate moderately complex audio."""
        # Combine multiple simple components
        component1 = self._generate_simple_speech(num_samples) if audio_type == "speech" else \
                    self._generate_simple_music(num_samples) if audio_type == "music" else \
                    self._generate_simple_environmental(num_samples)
        
        component2 = 0.3 * torch.randn(num_samples)  # Background noise
        
        # Add some amplitude modulation
        t = torch.linspace(0, num_samples / self.sample_rate, num_samples)
        modulation = 1 + 0.2 * torch.sin(2 * np.pi * 2 * t)  # 2Hz modulation
        
        return (component1 + component2) * modulation
        
    def _generate_complex_audio(self, num_samples: int, audio_type: str) -> torch.Tensor:
        """Generate complex multi-layered audio."""
        # Multiple overlapping components
        components = []
        
        # Main component
        components.append(self._generate_moderate_audio(num_samples, audio_type))
        
        # Secondary components
        for i in range(2):
            secondary_type = random.choice(["environmental", "speech", "music"])
            secondary = self._generate_simple_environmental(num_samples)
            components.append(0.3 * secondary)
            
        # Dynamic mixing
        t = torch.linspace(0, num_samples / self.sample_rate, num_samples)
        mix_weights = torch.softmax(torch.randn(len(components), 1) + 
                                   0.5 * torch.sin(2 * np.pi * 0.5 * t).unsqueeze(0), dim=0)
        
        complex_audio = sum(w * comp for w, comp in zip(mix_weights, components))
        return complex_audio.squeeze()


class ImageGenerator:
    """Generates realistic mock image data."""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
        
    def generate_image(self, complexity: str = "simple", 
                      image_type: str = "natural") -> torch.Tensor:
        """Generate mock image with specified characteristics.
        
        Args:
            complexity: Image complexity level
            image_type: Type of image (natural, objects, scenes)
            
        Returns:
            Image tensor of shape (3, H, W)
        """
        h, w = self.image_size
        
        if complexity == "simple":
            image = self._generate_simple_image(h, w, image_type)
        elif complexity == "moderate":
            image = self._generate_moderate_image(h, w, image_type)
        else:  # complex
            image = self._generate_complex_image(h, w, image_type)
            
        # Normalize to [0, 1] range
        image = torch.clamp(image, 0, 1)
        return image
        
    def _generate_simple_image(self, h: int, w: int, image_type: str) -> torch.Tensor:
        """Generate simple geometric patterns."""
        image = torch.zeros(3, h, w)
        
        if image_type == "objects":
            # Simple geometric shapes
            center_y, center_x = h // 2, w // 2
            radius = min(h, w) // 4
            
            # Create circle
            y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            distances = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            
            # Circle mask
            circle_mask = distances <= radius
            
            # Random color for the circle
            color = torch.rand(3)
            for c in range(3):
                image[c, circle_mask] = color[c]
            
        elif image_type == "scenes":
            # Simple gradient background
            gradient = torch.linspace(0, 1, h).unsqueeze(1).expand(-1, w)
            image[0] = gradient * 0.7  # Sky-like
            image[1] = gradient * 0.5
            image[2] = gradient * 0.9
            
        else:  # natural
            # Smooth color field
            x = torch.linspace(-1, 1, w)
            y = torch.linspace(-1, 1, h)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            
            image[0] = 0.5 + 0.3 * torch.sin(2 * X) * torch.cos(2 * Y)
            image[1] = 0.5 + 0.3 * torch.cos(3 * X) * torch.sin(Y)
            image[2] = 0.5 + 0.3 * torch.sin(X) * torch.sin(3 * Y)
            
        return image
        
    def _generate_moderate_image(self, h: int, w: int, image_type: str) -> torch.Tensor:
        """Generate moderately complex images."""
        # Start with simple image
        image = self._generate_simple_image(h, w, image_type)
        
        # Add texture
        noise = 0.1 * torch.randn(3, h, w)
        image += noise
        
        # Add some geometric features
        x = torch.linspace(-1, 1, w)
        y = torch.linspace(-1, 1, h)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Add stripes or patterns
        pattern = 0.1 * torch.sin(5 * X) * torch.sin(3 * Y)
        image += pattern.unsqueeze(0)
        
        return image
        
    def _generate_complex_image(self, h: int, w: int, image_type: str) -> torch.Tensor:
        """Generate complex multi-layered images."""
        # Multiple overlapping features
        image = torch.zeros(3, h, w)
        
        x = torch.linspace(-2, 2, w)
        y = torch.linspace(-2, 2, h)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Multiple frequency components
        for i in range(3):
            for j in range(3):
                freq_x, freq_y = (i + 1) * 2, (j + 1) * 2
                phase = torch.rand(1) * 2 * np.pi
                
                component = torch.sin(freq_x * X + freq_y * Y + phase)
                image[i % 3] += 0.1 * component
                
        # Add structured noise
        for c in range(3):
            noise = torch.randn(h // 4, w // 4)
            noise_upsampled = torch.nn.functional.interpolate(
                noise.unsqueeze(0).unsqueeze(0), 
                size=(h, w), 
                mode='bilinear'
            ).squeeze()
            image[c] += 0.2 * noise_upsampled
            
        # Base color
        base_colors = torch.rand(3, 1, 1) * 0.7
        image += base_colors
        
        return image


class TextGenerator:
    """Generates realistic mock text data."""
    
    def __init__(self):
        # Template patterns for different question types and difficulties
        self.templates = {
            QuestionType.AUDIO_DEPENDENT: {
                DifficultyLevel.EASY: [
                    "What sound do you hear?",
                    "Is there music playing?",
                    "What type of noise is this?",
                    "Can you hear any voices?"
                ],
                DifficultyLevel.MEDIUM: [
                    "How many different sounds can you identify?",
                    "What is the source of the primary sound?",
                    "Are there any background sounds besides the main one?",
                    "What emotion does this audio convey?"
                ],
                DifficultyLevel.HARD: [
                    "Analyze the acoustic properties and identify all sound sources in temporal order",
                    "What is the relationship between the audio content and the emotional context?",
                    "Describe the audio processing techniques that might have been applied",
                    "How does the audio quality suggest the recording environment?"
                ]
            },
            QuestionType.VISUAL_ONLY: {
                DifficultyLevel.EASY: [
                    "What color is the main object?",
                    "How many objects do you see?",
                    "What shape is in the center?",
                    "Is there a person in the image?"
                ],
                DifficultyLevel.MEDIUM: [
                    "What is the relationship between the objects in the scene?",
                    "Describe the lighting conditions in the image",
                    "What activity is taking place?",
                    "What time of day does this appear to be?"
                ],
                DifficultyLevel.HARD: [
                    "Analyze the compositional elements and their symbolic significance",
                    "What does the image suggest about the cultural or historical context?",
                    "How do the visual elements contribute to the overall narrative?",
                    "What artistic techniques are evident in this image?"
                ]
            },
            QuestionType.AUDIO_VISUAL: {
                DifficultyLevel.EASY: [
                    "Do the audio and visual elements match?",
                    "What is happening in both the audio and image?",
                    "Are the sound and picture from the same event?",
                    "What story do the audio and image tell together?"
                ],
                DifficultyLevel.MEDIUM: [
                    "How do the audio and visual elements complement each other?",
                    "What additional information does the audio provide about the scene?",
                    "Are there any contradictions between what you see and hear?",
                    "How does the audio enhance your understanding of the image?"
                ],
                DifficultyLevel.HARD: [
                    "Analyze the multimodal narrative structure and temporal relationships",
                    "How do the audio-visual elements create meaning beyond their individual contributions?",
                    "What implicit information emerges from the combination of modalities?",
                    "How might cultural context affect the interpretation of these combined elements?"
                ]
            }
        }
        
        self.answers = {
            "simple": [
                "Yes", "No", "Blue", "Red", "Circle", "Square", "One", "Two", "Music", "Voice",
                "Birds", "Traffic", "Water", "Wind", "Person", "Car", "House", "Tree"
            ],
            "moderate": [
                "The image shows a peaceful outdoor scene with natural lighting",
                "There are multiple sound sources including ambient and foreground elements",
                "The audio and visual elements are well synchronized",
                "This appears to be from a daytime recording in an urban environment",
                "The primary object has distinctive characteristics that suggest its function",
                "The soundscape includes both natural and artificial elements"
            ],
            "complex": [
                "The multimodal composition demonstrates sophisticated narrative techniques",
                "Analysis reveals complex interplay between temporal and spatial elements",
                "The audio-visual relationship suggests deeper symbolic meaning",
                "This represents a carefully constructed example of environmental storytelling",
                "The combination of modalities creates emergent semantic properties",
                "Cultural and contextual factors significantly influence interpretation"
            ]
        }
        
    def generate_qa_pair(self, question_type: QuestionType, difficulty: DifficultyLevel,
                        has_audio: bool = True, has_images: bool = True) -> Tuple[str, str]:
        """Generate a question-answer pair based on specifications.
        
        Args:
            question_type: Type of question to generate
            difficulty: Difficulty level
            has_audio: Whether sample has audio
            has_images: Whether sample has images
            
        Returns:
            Tuple of (question, answer)
        """
        # Adjust question type based on available modalities
        if not has_audio and question_type == QuestionType.AUDIO_DEPENDENT:
            question_type = QuestionType.VISUAL_ONLY
        elif not has_images and question_type == QuestionType.VISUAL_ONLY:
            question_type = QuestionType.TEXT_ONLY if not has_audio else QuestionType.AUDIO_DEPENDENT
            
        # Select template
        if question_type in self.templates and difficulty in self.templates[question_type]:
            question = random.choice(self.templates[question_type][difficulty])
        else:
            question = "What can you tell me about this sample?"
            
        # Generate appropriate answer
        if difficulty == DifficultyLevel.EASY:
            answer = random.choice(self.answers["simple"])
        elif difficulty == DifficultyLevel.MEDIUM:
            answer = random.choice(self.answers["moderate"])
        else:
            answer = random.choice(self.answers["complex"])
            
        return question, answer


class MockSAFEDataset:
    """Advanced mock dataset for SAFE testing with realistic characteristics."""
    
    def __init__(self, size: int = 1000, difficulty_distribution: Dict[DifficultyLevel, float] = None,
                 modality_distribution: Dict[str, float] = None, seed: int = 42):
        """Initialize mock SAFE dataset.
        
        Args:
            size: Number of samples to generate
            difficulty_distribution: Distribution of difficulty levels
            modality_distribution: Distribution of modality combinations
            seed: Random seed for reproducibility
        """
        self.size = size
        self.seed = seed
        
        # Set default distributions
        self.difficulty_distribution = difficulty_distribution or {
            DifficultyLevel.EASY: 0.4,
            DifficultyLevel.MEDIUM: 0.4,
            DifficultyLevel.HARD: 0.2
        }
        
        self.modality_distribution = modality_distribution or {
            "audio_visual": 0.4,
            "audio_only": 0.3,
            "visual_only": 0.25,
            "text_only": 0.05
        }
        
        # Initialize generators
        self.audio_gen = AudioGenerator()
        self.image_gen = ImageGenerator()
        self.text_gen = TextGenerator()
        
        # Pre-generate sample specifications for consistency
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.sample_specs = self._generate_sample_specs()
        
    def _generate_sample_specs(self) -> List[MockSampleSpec]:
        """Generate specifications for all samples."""
        specs = []
        
        for i in range(self.size):
            # Sample difficulty
            difficulty_probs = list(self.difficulty_distribution.values())
            difficulty = np.random.choice(list(self.difficulty_distribution.keys()), p=difficulty_probs)
            
            # Sample modality combination
            modality_probs = list(self.modality_distribution.values())
            modality = np.random.choice(list(self.modality_distribution.keys()), p=modality_probs)
            
            # Determine modalities and question type
            has_audio = modality in ["audio_visual", "audio_only"]
            has_images = modality in ["audio_visual", "visual_only"]
            
            if modality == "audio_visual":
                question_type = np.random.choice([
                    QuestionType.AUDIO_DEPENDENT,
                    QuestionType.VISUAL_ONLY,
                    QuestionType.AUDIO_VISUAL
                ], p=[0.4, 0.4, 0.2])
            elif modality == "audio_only":
                question_type = QuestionType.AUDIO_DEPENDENT
            elif modality == "visual_only":
                question_type = QuestionType.VISUAL_ONLY
            else:  # text_only
                question_type = QuestionType.TEXT_ONLY
                
            # Generate complexity levels
            complexity_map = {
                DifficultyLevel.EASY: "simple",
                DifficultyLevel.MEDIUM: "moderate",
                DifficultyLevel.HARD: "complex"
            }
            
            complexity = complexity_map[difficulty]
            
            spec = MockSampleSpec(
                question_type=question_type,
                difficulty=difficulty,
                has_audio=has_audio,
                has_images=has_images,
                audio_duration=np.random.uniform(5.0, 15.0),
                audio_complexity=complexity,
                image_complexity=complexity,
                question_complexity=complexity
            )
            
            specs.append(spec)
            
        return specs
        
    def __len__(self) -> int:
        return self.size
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index."""
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.size}")
            
        # Use consistent seed for each sample
        sample_seed = self.seed + idx
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        
        spec = self.sample_specs[idx]
        
        sample = {
            "question_type": spec.question_type.value,
            "difficulty": spec.difficulty.value,
            "sample_id": f"mock_{idx:06d}"
        }
        
        # Generate audio
        if spec.has_audio:
            audio_type = random.choice(["environmental", "speech", "music"])
            sample["audio"] = self.audio_gen.generate_audio(
                spec.audio_duration, 
                spec.audio_complexity, 
                audio_type
            )
        else:
            sample["audio"] = None
            
        # Generate images
        if spec.has_images:
            image_type = random.choice(["natural", "objects", "scenes"])
            sample["images"] = self.image_gen.generate_image(
                spec.image_complexity,
                image_type
            )  # Shape: (3, H, W)
        else:
            sample["images"] = None
            
        # Generate text
        question, answer = self.text_gen.generate_qa_pair(
            spec.question_type,
            spec.difficulty,
            spec.has_audio,
            spec.has_images
        )
        
        # Add image token for LLaVA compatibility when images are present
        if spec.has_images:
            question = f"<image>\n{question}"
        
        sample["question"] = question
        sample["answer"] = answer
        
        return sample
        
    def get_difficulty_distribution(self) -> Dict[str, int]:
        """Get actual difficulty distribution in the dataset."""
        distribution = defaultdict(int)
        for spec in self.sample_specs:
            distribution[spec.difficulty.value] += 1
        return dict(distribution)
        
    def get_modality_distribution(self) -> Dict[str, int]:
        """Get actual modality distribution in the dataset."""
        distribution = defaultdict(int)
        for spec in self.sample_specs:
            if spec.has_audio and spec.has_images:
                distribution["audio_visual"] += 1
            elif spec.has_audio:
                distribution["audio_only"] += 1
            elif spec.has_images:
                distribution["visual_only"] += 1
            else:
                distribution["text_only"] += 1
        return dict(distribution)
        
    def get_subset_by_difficulty(self, difficulty: DifficultyLevel) -> List[int]:
        """Get indices of samples with specified difficulty."""
        return [i for i, spec in enumerate(self.sample_specs) 
                if spec.difficulty == difficulty]
        
    def get_subset_by_modality(self, has_audio: bool = None, 
                              has_images: bool = None) -> List[int]:
        """Get indices of samples with specified modality combination."""
        indices = []
        for i, spec in enumerate(self.sample_specs):
            if has_audio is not None and spec.has_audio != has_audio:
                continue
            if has_images is not None and spec.has_images != has_images:
                continue
            indices.append(i)
        return indices


def create_curriculum_test_datasets() -> Dict[str, MockSAFEDataset]:
    """Create a set of test datasets for curriculum learning validation."""
    datasets = {}
    
    # Easy dataset - mostly simple samples
    datasets["easy"] = MockSAFEDataset(
        size=200,
        difficulty_distribution={
            DifficultyLevel.EASY: 0.8,
            DifficultyLevel.MEDIUM: 0.2,
            DifficultyLevel.HARD: 0.0
        },
        modality_distribution={
            "audio_visual": 0.3,
            "audio_only": 0.4,
            "visual_only": 0.3,
            "text_only": 0.0
        },
        seed=42
    )
    
    # Medium dataset - balanced difficulty
    datasets["medium"] = MockSAFEDataset(
        size=300,
        difficulty_distribution={
            DifficultyLevel.EASY: 0.2,
            DifficultyLevel.MEDIUM: 0.6,
            DifficultyLevel.HARD: 0.2
        },
        modality_distribution={
            "audio_visual": 0.5,
            "audio_only": 0.25,
            "visual_only": 0.2,
            "text_only": 0.05
        },
        seed=43
    )
    
    # Hard dataset - challenging samples
    datasets["hard"] = MockSAFEDataset(
        size=200,
        difficulty_distribution={
            DifficultyLevel.EASY: 0.0,
            DifficultyLevel.MEDIUM: 0.3,
            DifficultyLevel.HARD: 0.7
        },
        modality_distribution={
            "audio_visual": 0.6,
            "audio_only": 0.2,
            "visual_only": 0.15,
            "text_only": 0.05
        },
        seed=44
    )
    
    # Mixed dataset - realistic distribution
    datasets["mixed"] = MockSAFEDataset(
        size=500,
        difficulty_distribution={
            DifficultyLevel.EASY: 0.4,
            DifficultyLevel.MEDIUM: 0.4,
            DifficultyLevel.HARD: 0.2
        },
        modality_distribution={
            "audio_visual": 0.4,
            "audio_only": 0.3,
            "visual_only": 0.25,
            "text_only": 0.05
        },
        seed=45
    )
    
    return datasets