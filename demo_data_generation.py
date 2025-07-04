#!/usr/bin/env python3
"""
Demo script for generating yeast fermentation data with contamination events

This script demonstrates how to:
1. Generate individual fermentation episodes (fixed length)
2. Create datasets with various contamination scenarios
3. Visualize the fermentation profiles
4. Prepare data for RNN training with sliding window sequences

Updated for:
- Fixed episode lengths (48 hours)
- 1-hour time steps
- Persistent contamination (once it starts, it continues)
- Sliding window approach: multiple sequences per episode
- Configurable window size and stride for sequence generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fermentation_model import FermentationModel
from data_generator import FermentationDataGenerator
import os

def set_plotting_style():
    """Set up nice plotting style"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10


def demo_rnn_data_preparation():
    """Demonstrate data preparation for RNN training"""
    print("\n" + "=" * 60)
    print("DEMO: RNN Data Preparation")
    print("=" * 60)
    
    # Generate a dataset
    generator = FermentationDataGenerator()
    dataset = generator.generate_dataset(
        num_episodes=100,
        contamination_ratio=0.4,
        episode_length=48  # 48 hours
    )
    
    # Prepare RNN data using sliding windows
    print("\nPreparing data for RNN training with sliding windows...")
    window_size = 12  # Length of each sequence window
    stride = 1        # Step size for sliding window (larger stride = less overlap)
    
    # Example: For 48-hour episodes with window_size=24 and stride=6:
    # - Sequence 1: hours 0-23
    # - Sequence 2: hours 6-29
    # - Sequence 3: hours 12-35
    # - Sequence 4: hours 18-41
    # - Sequence 5: hours 24-47
    # Total: 5 sequences per episode
    
    print(f"Using window size: {window_size} time steps")
    print(f"Using stride: {stride} time steps")
    X, y, episode_ids = generator.prepare_rnn_data(dataset, window_size=window_size, stride=stride)
    
    print(f"\nRNN data shape:")
    print(f"  Input (X): {X.shape} [num_sequences, window_size, features]")
    print(f"  Output (y): {y.shape} [num_sequences, window_size]")
    print(f"  Episode IDs: {episode_ids.shape}")
    print(f"  Number of features: {X.shape[2]} (biomass, glucose, ethanol)")
    print(f"  Window size: {X.shape[1]} hours")
    print(f"  Total sequences: {X.shape[0]} (from {dataset['episode_id'].nunique()} episodes)")
    print(f"  Sequences per episode: ~{X.shape[0] / dataset['episode_id'].nunique():.1f}")
    print(f"  Contamination ratio: {y.mean():.3f}")
    
    # Split dataset for training
    print("\nSplitting dataset...")
    train_data, val_data, test_data = generator.split_dataset(dataset)
    
    # Prepare RNN data for each split
    X_train, y_train, _ = generator.prepare_rnn_data(train_data, window_size=window_size, stride=stride)
    X_val, y_val, _ = generator.prepare_rnn_data(val_data, window_size=window_size, stride=stride)
    X_test, y_test, _ = generator.prepare_rnn_data(test_data, window_size=window_size, stride=stride)
    
    print(f"\nRNN data splits:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"  Test: X={X_test.shape}, y={y_test.shape}")
    
    # Show contamination patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Show sample contamination patterns from different sequences
    sample_sequences = [0, 1, 2, 3]  # First 4 sequences
    for i, seq_idx in enumerate(sample_sequences):
        if seq_idx < len(y):
            time_hours = np.arange(len(y[seq_idx]))
            axes[i//2, i%2].plot(time_hours, y[seq_idx], 'r-', linewidth=2)
            axes[i//2, i%2].fill_between(time_hours, 0, 1, where=y[seq_idx]==1, alpha=0.3, color='red')
            axes[i//2, i%2].set_title(f'Sequence {seq_idx} from Episode {episode_ids[seq_idx]}')
            axes[i//2, i%2].set_xlabel('Time (h)')
            axes[i//2, i%2].set_ylabel('Contaminated')
            axes[i//2, i%2].set_ylim(-0.1, 1.1)
            axes[i//2, i%2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Sample Sliding Window Sequences for RNN Training', fontsize=16, y=1.02)
    plt.show()
    
    # Save RNN-ready data
    print("\nSaving RNN-ready data...")
    np.save("rnn_train_X.npy", X_train)
    np.save("rnn_train_y.npy", y_train)
    np.save("rnn_val_X.npy", X_val)
    np.save("rnn_val_y.npy", y_val)
    np.save("rnn_test_X.npy", X_test)
    np.save("rnn_test_y.npy", y_test)
    
    # Save complete dataset as CSV
    print("\nSaving complete dataset...")
    dataset.to_csv("complete_fermentation_dataset.csv", index=False)
    
    # Save episode summaries
    episode_summaries = generator.get_episode_summary(dataset)
    episode_summaries.to_csv("episode_summaries.csv", index=False)
    
    print("\nFiles saved:")
    print("  - rnn_train_X.npy, rnn_train_y.npy")
    print("  - rnn_val_X.npy, rnn_val_y.npy")
    print("  - rnn_test_X.npy, rnn_test_y.npy")
    print("  - complete_fermentation_dataset.csv (all episodes and features)")
    print("  - episode_summaries.csv")
    
    # Show dataset structure
    print(f"\nComplete dataset structure:")
    print(f"  Shape: {dataset.shape}")
    print(f"  Columns: {list(dataset.columns)}")
    print(f"  Episodes: {dataset['episode_id'].nunique()}")
    print(f"  Time steps per episode: {len(dataset[dataset['episode_id'] == 0])}")
    
    return X, y, episode_ids

def main():
    """Run all demonstrations"""
    print("Yeast Fermentation Episode Generation Demo")
    print("=" * 60)
    print("Updated for sliding window sequence generation from episodes")
    print("Each step represents 1 hour of fermentation time")
    print("Multiple sequences are generated from each episode using sliding windows")
    
    # Set up plotting
    set_plotting_style()
    
    # Run demonstrations
    try:
        # Demo 4: RNN data preparation
        demo_rnn_data_preparation()
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure all required packages are installed:")
        print("  uv add numpy pandas scipy matplotlib seaborn")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 