# Yeast Fermentation Episode Generation

A comprehensive system for generating synthetic yeast fermentation data with contamination events, designed for RNN contamination detection models.

## Features

- **Fixed Episode Structure**: Episodes have consistent length (48 hours) with 1-hour time steps
- **Persistent Contamination**: Once contamination starts, it continues until episode end
- **Multiple Contamination Types**: Bacterial, wild yeast, and acidic contamination
- **RNN-Ready Data**: Direct compatibility with sequence-to-sequence models
- **Simple Features**: 3 basic measurements for streamlined training

## Quick Start

```bash
# Install dependencies
uv add numpy pandas scipy matplotlib seaborn

# Run quick start demo
python quick_start.py

# Run comprehensive demo
python demo_data_generation.py
```

## System Overview

### Episode Structure
- **Length**: 48 hours (48 time steps)
- **Time Resolution**: 1 hour per step
- **Contamination**: Randomly starts between 10-80% through episode
- **Persistence**: Once contamination begins, it continues to episode end

### Data Format
Each episode generates a DataFrame with:
- `step`: Time step (0-47)
- `time_hours`: Time in hours (0-47)
- `biomass`: Biomass concentration (g/L)
- `glucose`: Glucose concentration (g/L)
- `ethanol`: Ethanol concentration (g/L)
- `contaminated`: Boolean contamination flag
- `contamination_type`: Type of contamination

## Code Examples

### Generate Single Episode

```python
from fermentation_model import FermentationModel
from data_generator import FermentationDataGenerator

# Create model and generator
model = FermentationModel()
generator = FermentationDataGenerator(model)

# Generate clean episode
clean_episode = generator.generate_single_episode(
    episode_length=48,  # 48 hours
    contamination_probability=0.0,
    episode_id=0
)

# Generate contaminated episode
contaminated_episode = generator.generate_single_episode(
    episode_length=48,
    contamination_probability=1.0,  # Force contamination
    episode_id=1
)
```

### Generate Dataset

```python
# Generate dataset with multiple episodes
dataset = generator.generate_dataset(
    num_episodes=100,
    contamination_ratio=0.3,  # 30% contaminated
    episode_length=48,
    save_to_file="fermentation_episodes.csv",  # Basic CSV
    save_complete_dataset=True  # Saves complete dataset automatically
)

# This creates two files:
# 1. fermentation_episodes.csv (basic format)
# 2. complete_fermentation_dataset_100episodes.csv (all features)
```

### Complete Dataset Structure

The complete CSV dataset contains all episodes with the following columns:

**Time & Episode Info:**
- `step`: Time step (0-47)
- `time_hours`: Time in hours (0-47)  
- `episode_id`: Unique episode identifier

**Measurements (RNN Input Features):**
- `biomass`: Biomass concentration (g/L)
- `glucose`: Glucose concentration (g/L)
- `ethanol`: Ethanol concentration (g/L)

**Contamination Info (Labels & Metadata):**
- `contaminated`: Boolean contamination status (RNN target)
- `contamination_type`: Type of contamination ('bacterial', 'wild_yeast', 'acidic', 'none')
- `has_contamination`: Episode-level contamination flag
- `contamination_step`: Step when contamination started (-1 if none)

**Example CSV structure:**
```
step,time_hours,biomass,glucose,ethanol,contaminated,contamination_type,episode_id,has_contamination,contamination_step
0,0,0.5,20.0,0.0,False,none,0,False,-1
1,1,0.52,19.8,0.15,False,none,0,False,-1
...
15,15,1.2,15.2,2.8,True,bacterial,1,True,15
16,16,1.25,14.8,3.1,True,bacterial,1,True,15
```

### RNN Data Preparation

```python
# Prepare data for RNN training
X, y, episode_ids = generator.prepare_rnn_data(dataset)
print(f"X shape: {X.shape}")  # [num_episodes, 48, 3]
print(f"y shape: {y.shape}")  # [num_episodes, 48]

# Split dataset
train_data, val_data, test_data = generator.split_dataset(dataset)
```

## Model Details

### Fermentation Kinetics
Based on Monod kinetics with ethanol inhibition:
- Biomass growth with glucose consumption
- Ethanol production from glucose
- Death rate and maintenance effects

### Contamination Models
- **Bacterial**: Altered yield coefficients, different metabolic pathways
- **Wild Yeast**: Modified kinetic parameters, different substrate affinities
- **Acidic**: Reduced overall activity, pH stress effects

### Features
The system generates 3 basic features per time step:
1. `biomass`: Biomass concentration (g/L)
2. `glucose`: Glucose concentration (g/L)
3. `ethanol`: Ethanol concentration (g/L)

## File Structure

```
yeast-fermentation/
├── fermentation_model.py      # Core fermentation model
├── data_generator.py          # Data generation utilities  
├── demo_data_generation.py    # Comprehensive demo
├── quick_start.py            # Quick start script
├── pyproject.toml            # Project dependencies
└── README.md                 # This file
```

## Advanced Usage

### Custom Parameters

```python
# Custom fermentation parameters
custom_params = {
    'mu_max': 0.5,     # Maximum specific growth rate
    'K_s': 0.3,        # Half-saturation constant
    'Y_xs': 0.6,       # Biomass yield coefficient
    'Y_ps': 0.4,       # Ethanol yield coefficient
    'K_i': 100.0,      # Ethanol inhibition constant
    'k_d': 0.01,       # Death rate constant
    'alpha': 0.15      # Non-growth ethanol production
}

model = FermentationModel(custom_params)
```

### Episode Length Options

```python
# Available episode lengths
episode_options = FermentationModel.get_episode_length_options()
print(episode_options)
# {'24_hours': 24, '48_hours': 48, '72_hours': 72, '96_hours': 96}
```

### Contamination Scenarios

```python
# Force specific contamination type
episode = model.simulate_episode(
    initial_conditions=[0.5, 20.0, 0.0],
    episode_length=48,
    contamination_step=12,  # Start at 12 hours
    contamination_type='bacterial'
)
```

## RNN Training Notes

The data is structured for sequence-to-sequence RNN models:
- **Input**: `[batch_size, 48, 3]` - 48 time steps, 3 features
- **Output**: `[batch_size, 48]` - 48 contamination labels
- **Task**: Binary classification at each time step

Recommended architectures:
- LSTM/GRU with 32-128 hidden units
- Bidirectional RNNs for better context
- Attention mechanisms for long sequences

## Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scipy`: Scientific computing (ODE solver)
- `matplotlib`: Plotting
- `seaborn`: Statistical visualizations

## Installation

```bash
# Using uv (recommended)
uv add numpy pandas scipy matplotlib seaborn

# Using pip
pip install numpy pandas scipy matplotlib seaborn
```

## Key Improvements

This version features:
1. **Fixed Episode Lengths**: All episodes are exactly 48 hours (48 steps)
2. **1-Hour Time Steps**: Simplified time resolution for easier RNN training
3. **Persistent Contamination**: Once contamination starts, it continues
4. **RNN-Ready Format**: Direct compatibility with sequence models
5. **Simple Features**: Only 3 basic measurements for streamlined training

The system is optimized for training contamination detection RNNs with clean, consistent data structure and realistic fermentation dynamics. 