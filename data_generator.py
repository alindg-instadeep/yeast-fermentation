import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import random
from fermentation_model import FermentationModel


class FermentationDataGenerator:
    """
    Generate synthetic fermentation data with fixed episode lengths for RNN training
    
    Updated for:
    - Fixed episode lengths (default: 48 hours)
    - 1-hour time steps
    - Persistent contamination (once it starts, it continues to the end)
    """
    
    def __init__(self, model: Optional[FermentationModel] = None):
        """
        Initialize the data generator
        
        Args:
            model: FermentationModel instance, creates default if None
        """
        self.model = model if model is not None else FermentationModel()
        self.contamination_types = ['bacterial', 'wild_yeast', 'acidic']
        self.episode_length = 48
        
    def generate_contamination_event(self, episode_length: int, 
                                   contamination_probability: float = 0.3) -> Tuple[Optional[int], str]:
        """
        Generate a single contamination event for an episode
        
        Args:
            episode_length: Length of episode in hours
            contamination_probability: Probability of contamination occurring
            
        Returns:
            Tuple of (contamination_step, contamination_type)
            contamination_step is None if no contamination occurs
        """
        if random.random() < contamination_probability:
            # Contamination can start anywhere from 10% to 80% through the episode
            contamination_step = random.randint(
                int(episode_length * 0.1), 
                int(episode_length * 0.8)
            )
            contamination_type = random.choice(self.contamination_types)
            return contamination_step, contamination_type
        else:
            return None, 'none'
    
    def generate_initial_conditions(self, variation: float = 0.2) -> List[float]:
        """
        Generate random initial conditions with variation
        
        Args:
            variation: Fractional variation in initial conditions
            
        Returns:
            [X0, S0, P0] - Initial biomass, glucose, ethanol concentrations
        """
        # Base initial conditions
        base_conditions = [0.5, 20.0, 0.0]  # biomass, glucose, ethanol
        
        # Add variation
        varied_conditions = []
        for i, base_val in enumerate(base_conditions):
            if i == 2:  # Ethanol starts at 0
                varied_conditions.append(0.0)
            else:
                variation_factor = 1.0 + random.uniform(-variation, variation)
                varied_conditions.append(max(0.1, base_val * variation_factor))
        
        return varied_conditions
    
    def generate_fermentation_parameters(self, variation: float = 0.15) -> Dict:
        """
        Generate random fermentation parameters with variation
        
        Args:
            variation: Fractional variation in parameters
            
        Returns:
            Dictionary of varied parameters
        """
        base_params = {
            'mu_max': 0.4,
            'K_s': 0.5,
            'Y_xs': 0.5,
            'Y_ps': 0.45,
            'K_i': 80.0,
            'k_d': 0.02,
            'alpha': 0.1
        }
        
        varied_params = {}
        for key, base_val in base_params.items():
            variation_factor = 1.0 + random.uniform(-variation, variation)
            varied_params[key] = max(0.01, base_val * variation_factor)
        
        return varied_params
    
    def generate_single_episode(self, 
                               episode_length: int = 48,
                               contamination_probability: float = 1.0,
                               parameter_variation: float = 0.15,
                               initial_variation: float = 0.2,
                               episode_id: int = 0) -> pd.DataFrame:
        """
        Generate a single fermentation episode with potential contamination
        
        Args:
            episode_length: Length of episode in hours (default: 48 hours)
            contamination_probability: Probability of contamination
            parameter_variation: Variation in kinetic parameters
            initial_variation: Variation in initial conditions
            episode_id: ID for this episode
            
        Returns:
            DataFrame with episode data
        """
        # Generate varied parameters and initial conditions
        params = self.generate_fermentation_parameters(parameter_variation)
        initial_conditions = self.generate_initial_conditions(initial_variation)
        
        # Create model with varied parameters
        model = FermentationModel(params)
        
        # Generate contamination event
        contamination_step, contamination_type = self.generate_contamination_event(
            episode_length, contamination_probability
        )
        
        # Simulate episode
        data = model.simulate_episode(
            initial_conditions, 
            episode_length, 
            contamination_step, 
            contamination_type
        )
        
        # Add episode metadata
        data['episode_id'] = episode_id
        data['has_contamination'] = contamination_step is not None
        data['contamination_step'] = contamination_step if contamination_step is not None else -1
        
        return data
    
    def generate_dataset(self, 
                        num_episodes: int = 20,
                        contamination_ratio: float = 0.5,
                        episode_length: int = 48,
                        parameter_variation: float = 0.15,
                        initial_variation: float = 0.2,
                        save_to_file: Optional[str] = None,) -> pd.DataFrame:
        """
        Generate a complete dataset of fermentation episodes
        
        Args:
            num_episodes: Number of fermentation episodes to generate
            contamination_ratio: Fraction of episodes with contamination
            episode_length: Length of each episode in hours (default: 48 hours)
            parameter_variation: Variation in kinetic parameters
            initial_variation: Variation in initial conditions
            save_to_file: Optional filename to save basic dataset
            save_complete_dataset: Whether to save complete dataset with all features
            
        Returns:
            DataFrame with all episode data
        """
        
        print(f"Generating {num_episodes} fermentation episodes...")
        print(f"Episode length: {episode_length} hours")
        
        all_data = []
        contaminated_count = 0
        target_contaminated = int(num_episodes * contamination_ratio)
        
        # Pre-determine which episodes should be contaminated for better distribution
        episodes_to_contaminate = set(random.sample(range(num_episodes), target_contaminated))
        
        for episode_id in range(num_episodes):
            # Determine if this episode should be contaminated
            if episode_id in episodes_to_contaminate:
                contamination_prob = 1.0  # Guarantee contamination
            else:
                contamination_prob = 0.0  # No contamination
            
            # Generate episode data
            episode_data = self.generate_single_episode(
                episode_length=episode_length,
                contamination_probability=contamination_prob,
                parameter_variation=parameter_variation,
                initial_variation=initial_variation,
                episode_id=episode_id
            )
            
            # Update contamination count
            if episode_data['has_contamination'].any():
                contaminated_count += 1
            
            all_data.append(episode_data)
            
            # Progress update
            if (episode_id + 1) % 10 == 0:
                print(f"Generated {episode_id + 1}/{num_episodes} episodes...")
        
        # Combine all data
        dataset = pd.concat(all_data, ignore_index=True)
        
        print(f"Dataset generation complete!")
        print(f"Total episodes: {num_episodes}")
        print(f"Contaminated episodes: {contaminated_count}")
        print(f"Clean episodes: {num_episodes - contaminated_count}")
        print(f"Total data points: {len(dataset)}")
        
        # Save to file if requested
        if save_to_file:
            dataset.to_csv(save_to_file, index=False)
            print(f"Dataset saved to {save_to_file}")
        
        return dataset
    
    def prepare_rnn_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for RNN training - each episode becomes one sequence
        
        Args:
            data: Fermentation dataset
            
        Returns:
            Tuple of (X, y, episode_ids) arrays
            - X: Input sequences [num_episodes, episode_length, num_features]
            - y: Target sequences [num_episodes, episode_length] (contamination labels)
            - episode_ids: Array of episode IDs for reference
        """
        # Feature columns for RNN - only observable measurements
        # (contamination status is the target, not an input!)
        feature_columns = [
            'biomass', 'glucose', 'ethanol'
        ]
        
        X_episodes = []
        y_episodes = []
        episode_ids = []
        
        # Process each episode separately
        for episode_id in sorted(data['episode_id'].unique()):
            episode_data = data[data['episode_id'] == episode_id].copy()
            
            # Extract features and labels
            X_episode = episode_data[feature_columns].values
            y_episode = episode_data['contaminated'].values.astype(int)
            
            X_episodes.append(X_episode)
            y_episodes.append(y_episode)
            episode_ids.append(episode_id)
        
        return np.array(X_episodes), np.array(y_episodes), np.array(episode_ids)
    
    def get_episode_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for each episode
        
        Args:
            data: Fermentation dataset
            
        Returns:
            DataFrame with episode-level summaries
        """
        summaries = []
        
        for episode_id in data['episode_id'].unique():
            episode_data = data[data['episode_id'] == episode_id]
            
            summary = {
                'episode_id': episode_id,
                'episode_length': len(episode_data),
                'has_contamination': episode_data['has_contamination'].any(),
                'contamination_step': episode_data['contamination_step'].iloc[0] if episode_data['has_contamination'].any() else -1,
                'contamination_duration': episode_data['contaminated'].sum(),  # hours
                'contamination_fraction': episode_data['contaminated'].mean(),
                'final_biomass': episode_data['biomass'].iloc[-1],
                'final_glucose': episode_data['glucose'].iloc[-1],
                'final_ethanol': episode_data['ethanol'].iloc[-1],
                'max_biomass': episode_data['biomass'].max(),
                'max_ethanol': episode_data['ethanol'].max(),
                'glucose_consumed': episode_data['glucose'].iloc[0] - episode_data['glucose'].iloc[-1],
                'ethanol_yield': episode_data['ethanol'].iloc[-1] / (episode_data['glucose'].iloc[0] - episode_data['glucose'].iloc[-1] + 1e-6),
                'contamination_type': episode_data['contamination_type'].iloc[-1] if episode_data['contaminated'].any() else 'none'
            }
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def split_dataset(self, data: pd.DataFrame, 
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, 
                     test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/validation/test sets by episodes
        
        Args:
            data: Complete fermentation dataset
            train_ratio: Fraction of episodes for training
            val_ratio: Fraction of episodes for validation
            test_ratio: Fraction of episodes for testing
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        # Get unique episode IDs
        episode_ids = sorted(data['episode_id'].unique())
        num_episodes = len(episode_ids)
        
        # Shuffle episode IDs
        random.shuffle(episode_ids)
        
        # Split episode IDs
        train_size = int(num_episodes * train_ratio)
        val_size = int(num_episodes * val_ratio)
        
        train_episodes = episode_ids[:train_size]
        val_episodes = episode_ids[train_size:train_size + val_size]
        test_episodes = episode_ids[train_size + val_size:]
        
        # Split data
        train_data = data[data['episode_id'].isin(train_episodes)]
        val_data = data[data['episode_id'].isin(val_episodes)]
        test_data = data[data['episode_id'].isin(test_episodes)]
        
        print(f"Dataset split:")
        print(f"  Train: {len(train_episodes)} episodes, {len(train_data)} data points")
        print(f"  Validation: {len(val_episodes)} episodes, {len(val_data)} data points")
        print(f"  Test: {len(test_episodes)} episodes, {len(test_data)} data points")
        
        return train_data, val_data, test_data 