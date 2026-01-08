"""
Asian Region Audio Dataset Generator for ALM Training

This module creates comprehensive audio datasets with speech and non-speech samples
for Asian languages including Mandarin, Urdu, Hindi, Telugu, Tamil, Bangla, and English.
"""

import os
import json
import random
import librosa
import soundfile as sf
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioSample:
    """Data class for audio samples"""
    audio_id: str
    instruction: str
    output: str
    dataset: str
    task: str
    language: str
    duration: float
    sample_rate: int = 16000

class AsianAudioDatasetGenerator:
    """Generator for Asian region audio datasets"""
    
    def __init__(self, output_dir: str = "datasets/asian_audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Asian languages configuration
        self.languages = {
            'mandarin': {'code': 'zh', 'name': 'Mandarin Chinese'},
            'urdu': {'code': 'ur', 'name': 'Urdu'},
            'hindi': {'code': 'hi', 'name': 'Hindi'},
            'telugu': {'code': 'te', 'name': 'Telugu'},
            'tamil': {'code': 'ta', 'name': 'Tamil'},
            'bangla': {'code': 'bn', 'name': 'Bangla'},
            'english': {'code': 'en', 'name': 'English'}
        }
        
        # Audio event categories for non-speech sounds
        self.audio_events = {
            'transportation': [
                'car_engine', 'car_horn', 'motorcycle', 'bus', 'train', 'airplane',
                'helicopter', 'boat', 'subway', 'truck', 'ambulance', 'police_siren'
            ],
            'environmental': [
                'rain', 'thunder', 'wind', 'ocean_waves', 'birds_chirping',
                'dog_barking', 'cat_meowing', 'cow_mooing', 'rooster_crowing'
            ],
            'urban': [
                'construction', 'traffic', 'crowd_chatter', 'footsteps',
                'door_slam', 'phone_ringing', 'alarm_clock', 'doorbell'
            ],
            'music': [
                'piano', 'guitar', 'violin', 'drums', 'flute', 'trumpet',
                'singing', 'orchestra', 'rock_music', 'classical_music'
            ],
            'household': [
                'vacuum_cleaner', 'washing_machine', 'microwave', 'refrigerator',
                'dishwasher', 'air_conditioner', 'fan', 'blender'
            ]
        }
        
        # Complex reasoning scenarios
        self.scenarios = {
            'airport': {
                'description': 'Person at airport with various announcements and sounds',
                'audio_events': ['airplane', 'crowd_chatter', 'announcement', 'footsteps'],
                'reasoning_tasks': [
                    'What can you infer about the person\'s location?',
                    'What time of day might it be based on the sounds?',
                    'What type of announcements are being made?',
                    'What is the mood of the environment?'
                ]
            },
            'restaurant': {
                'description': 'Busy restaurant with multiple conversations and kitchen sounds',
                'audio_events': ['crowd_chatter', 'dishes_clinking', 'cooking', 'music'],
                'reasoning_tasks': [
                    'How many people are likely in the restaurant?',
                    'What type of cuisine is being prepared?',
                    'What is the atmosphere like?',
                    'What languages are being spoken?'
                ]
            },
            'street_market': {
                'description': 'Vibrant street market with vendors and customers',
                'audio_events': ['crowd_chatter', 'vehicle_traffic', 'vendor_calls', 'music'],
                'reasoning_tasks': [
                    'What type of market is this?',
                    'What are vendors selling?',
                    'What is the cultural context?',
                    'What time of day is it?'
                ]
            },
            'home_environment': {
                'description': 'Home environment with family activities',
                'audio_events': ['conversation', 'tv_sound', 'cooking', 'children_playing'],
                'reasoning_tasks': [
                    'What activities are happening at home?',
                    'How many family members are present?',
                    'What is the emotional atmosphere?',
                    'What time of day is it?'
                ]
            }
        }
    
    def generate_synthetic_audio(self, duration: float, sample_rate: int = 16000) -> np.ndarray:
        """Generate synthetic audio for testing purposes"""
        # Generate white noise as base
        audio = np.random.normal(0, 0.1, int(duration * sample_rate))
        
        # Add some frequency components to make it more realistic
        t = np.linspace(0, duration, int(duration * sample_rate))
        audio += 0.1 * np.sin(2 * np.pi * 440 * t)  # A4 note
        audio += 0.05 * np.sin(2 * np.pi * 880 * t)  # A5 note
        
        return audio
    
    def create_audio_sample(self, scenario: str, language: str, task_type: str) -> AudioSample:
        """Create a single audio sample with instruction and output"""
        scenario_info = self.scenarios[scenario]
        
        # Generate instruction based on task type
        if task_type == 'speech_recognition':
            instruction = f"Transcribe the speech in this {language} audio recording."
            output = f"This audio contains speech in {self.languages[language]['name']}. The speaker is discussing {scenario}."
        
        elif task_type == 'audio_event_detection':
            events = random.sample(scenario_info['audio_events'], min(3, len(scenario_info['audio_events'])))
            instruction = "Identify all the audio events and sounds in this recording."
            output = f"The audio contains the following sounds: {', '.join(events)}."
        
        elif task_type == 'complex_reasoning':
            reasoning_task = random.choice(scenario_info['reasoning_tasks'])
            instruction = reasoning_task
            output = f"Based on the audio analysis: {self._generate_reasoning_output(scenario, reasoning_task)}"
        
        elif task_type == 'multilingual_understanding':
            instruction = f"Analyze this audio and identify the languages spoken and the overall context."
            output = f"The audio contains speech in {self.languages[language]['name']} and English. The context is {scenario}."
        
        else:
            instruction = "Analyze this audio and provide a comprehensive description."
            output = f"This audio recording captures a {scenario} environment with various sounds and speech in {self.languages[language]['name']}."
        
        # Generate unique audio ID
        audio_id = f"{scenario}_{language}_{task_type}_{random.randint(1000, 9999)}"
        
        return AudioSample(
            audio_id=audio_id,
            instruction=instruction,
            output=output,
            dataset="asian_audio_dataset",
            task=task_type,
            language=language,
            duration=random.uniform(5.0, 30.0)  # Random duration between 5-30 seconds
        )
    
    def _generate_reasoning_output(self, scenario: str, reasoning_task: str) -> str:
        """Generate reasoning output based on scenario and task"""
        reasoning_outputs = {
            'airport': {
                'location': 'The person is at an airport, specifically in the boarding area or waiting hall based on airplane sounds and announcements.',
                'time': 'It appears to be daytime based on the level of activity and announcements.',
                'announcements': 'The announcements are about flight schedules and boarding information.',
                'mood': 'The environment is busy but organized, with people moving around.'
            },
            'restaurant': {
                'people': 'There are approximately 15-20 people in the restaurant based on the chatter level.',
                'cuisine': 'The cooking sounds suggest Asian cuisine preparation.',
                'atmosphere': 'The restaurant has a lively and social atmosphere.',
                'languages': 'Multiple languages are being spoken, indicating a diverse clientele.'
            },
            'street_market': {
                'market_type': 'This appears to be a traditional street market with local vendors.',
                'selling': 'Vendors are selling various goods, likely food and local products.',
                'cultural': 'The market reflects local cultural traditions and practices.',
                'time': 'It appears to be morning or early afternoon based on activity level.'
            },
            'home_environment': {
                'activities': 'Family members are engaged in cooking, watching TV, and children are playing.',
                'members': 'There are at least 3-4 family members present.',
                'atmosphere': 'The home has a warm and comfortable family atmosphere.',
                'time': 'This appears to be evening time based on the activities.'
            }
        }
        
        # Extract key from reasoning task
        for key in reasoning_outputs[scenario]:
            if key in reasoning_task.lower():
                return reasoning_outputs[scenario][key]
        
        return f"The audio analysis suggests this is a {scenario} environment with typical associated sounds and activities."
    
    def generate_dataset(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """Generate the complete dataset"""
        logger.info(f"Generating {num_samples} audio samples for Asian region dataset...")
        
        dataset = []
        task_types = ['speech_recognition', 'audio_event_detection', 'complex_reasoning', 'multilingual_understanding']
        
        for i in range(num_samples):
            # Randomly select scenario, language, and task type
            scenario = random.choice(list(self.scenarios.keys()))
            language = random.choice(list(self.languages.keys()))
            task_type = random.choice(task_types)
            
            # Create audio sample
            sample = self.create_audio_sample(scenario, language, task_type)
            
            # Convert to dictionary format
            sample_dict = {
                'audio_id': sample.audio_id,
                'instruction': sample.instruction,
                'output': sample.output,
                'dataset': sample.dataset,
                'task': sample.task,
                'language': sample.language,
                'duration': sample.duration,
                'sample_rate': sample.sample_rate
            }
            
            dataset.append(sample_dict)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples...")
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str = "asian_audio_dataset.json"):
        """Save dataset to JSON file"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Total samples: {len(dataset)}")
        
        # Print dataset statistics
        self.print_dataset_stats(dataset)
    
    def print_dataset_stats(self, dataset: List[Dict[str, Any]]):
        """Print dataset statistics"""
        logger.info("\n=== Dataset Statistics ===")
        
        # Language distribution
        language_counts = {}
        for sample in dataset:
            lang = sample['language']
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        logger.info("Language distribution:")
        for lang, count in language_counts.items():
            logger.info(f"  {self.languages[lang]['name']}: {count} samples")
        
        # Task distribution
        task_counts = {}
        for sample in dataset:
            task = sample['task']
            task_counts[task] = task_counts.get(task, 0) + 1
        
        logger.info("\nTask distribution:")
        for task, count in task_counts.items():
            logger.info(f"  {task}: {count} samples")
        
        # Duration statistics
        durations = [sample['duration'] for sample in dataset]
        logger.info(f"\nDuration statistics:")
        logger.info(f"  Average: {np.mean(durations):.2f} seconds")
        logger.info(f"  Min: {np.min(durations):.2f} seconds")
        logger.info(f"  Max: {np.max(durations):.2f} seconds")

def main():
    """Main function to generate the dataset"""
    generator = AsianAudioDatasetGenerator()
    
    # Generate training dataset
    logger.info("Generating training dataset...")
    train_dataset = generator.generate_dataset(num_samples=2000)
    generator.save_dataset(train_dataset, "train_asian_audio_dataset.json")
    
    # Generate validation dataset
    logger.info("Generating validation dataset...")
    val_dataset = generator.generate_dataset(num_samples=500)
    generator.save_dataset(val_dataset, "val_asian_audio_dataset.json")
    
    # Generate test dataset
    logger.info("Generating test dataset...")
    test_dataset = generator.generate_dataset(num_samples=300)
    generator.save_dataset(test_dataset, "test_asian_audio_dataset.json")
    
    logger.info("Dataset generation completed!")

if __name__ == "__main__":
    main()
