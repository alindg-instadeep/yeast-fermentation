import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pandas as pd

class FermentationModel:
    """
    A comprehensive yeast fermentation model based on Monod kinetics
    Models glucose consumption, ethanol production, and biomass growth
    
    Updated for fixed episode lengths with 1-minute steps and persistent contamination
    """
    
    def __init__(self, parameters: Optional[Dict] = None):
        """
        Initialize fermentation model with kinetic parameters
        
        Parameters:
        - mu_max: Maximum specific growth rate (1/h)
        - K_s: Half-saturation constant for glucose (g/L)
        - Y_xs: Yield coefficient biomass/glucose (g/g)
        - Y_ps: Yield coefficient ethanol/glucose (g/g)
        - K_i: Inhibition constant for ethanol (g/L)
        - k_d: Death rate constant (1/h)
        - alpha: Non-growth associated ethanol production rate
        """
        
        default_params = {
            'mu_max': 0.4,      # Maximum specific growth rate (1/h)
            'K_s': 0.5,         # Half-saturation constant for glucose (g/L)
            'Y_xs': 0.5,        # Yield coefficient biomass/glucose (g/g)
            'Y_ps': 0.45,       # Yield coefficient ethanol/glucose (g/g)
            'K_i': 80.0,        # Inhibition constant for ethanol (g/L)
            'k_d': 0.02,        # Death rate constant (1/h)
            'alpha': 0.1        # Non-growth associated ethanol production
        }
        
        self.params = default_params if parameters is None else {**default_params, **parameters}
        
    def fermentation_dynamics(self, state: List[float], t: float) -> List[float]:
        """
        Differential equations for normal yeast fermentation
        
        State variables:
        - X: Biomass concentration (g/L)
        - S: Glucose concentration (g/L)
        - P: Ethanol concentration (g/L)
        
        Differential equations:
        dX/dt = μX - k_d*X
        dS/dt = -μX/Y_xs - αX
        dP/dt = μX/Y_ps + αX
        
        Where:
        - μ = μ_max * (S/(K_s + S)) * (K_i/(K_i + P))  # Monod kinetics with ethanol inhibition
        - μ_max: Maximum specific growth rate
        - K_s: Half-saturation constant for glucose
        - K_i: Inhibition constant for ethanol
        - Y_xs: Yield coefficient biomass/glucose
        - Y_ps: Yield coefficient ethanol/glucose
        - k_d: Death rate constant
        - α: Non-growth associated ethanol production rate
        """
        X, S, P = state
        
        # Ensure non-negative concentrations for calculations
        X = max(0.0, X)
        S = max(0.0, S)
        P = max(0.0, P)
        
        # Monod kinetics with ethanol inhibition
        # When glucose is very low, growth stops
        if S < 1e-6:  # Essentially no glucose left
            mu = 0.0
        else:
            mu = self.params['mu_max'] * (S / (self.params['K_s'] + S)) * (self.params['K_i'] / (self.params['K_i'] + P))
        
        # Rate equations
        dX_dt = mu * X - self.params['k_d'] * X
        
        # Glucose consumption - stop when glucose is depleted
        if S < 1e-6:
            dS_dt = 0.0
        else:
            dS_dt = -mu * X / self.params['Y_xs'] - self.params['alpha'] * X
            # Ensure glucose consumption doesn't exceed available glucose
            max_consumption = S / (1/60)  # Maximum consumption per hour
            dS_dt = max(dS_dt, -max_consumption)
        
        # Ethanol production
        dP_dt = mu * X / self.params['Y_ps'] + self.params['alpha'] * X
        
        return [dX_dt, dS_dt, dP_dt]
    
    def contaminated_dynamics(self, state: List[float], contamination_type: str = 'bacterial') -> List[float]:
        """
        Differential equations for contaminated fermentation
        
        Contamination types:
        - 'bacterial': Bacterial contamination (different metabolic pathways)
        - 'wild_yeast': Wild yeast contamination
        - 'acidic': Acidic contamination affecting pH
        """
        X, S, P = state
        
        # Ensure non-negative concentrations for calculations
        X = max(0.0, X)
        S = max(0.0, S)
        P = max(0.0, P)
        
        if contamination_type == 'bacterial':
            # Bacterial contamination: Lower ethanol yield, higher biomass yield
            if S < 1e-6:
                mu = 0.0
            else:
                mu = self.params['mu_max'] * 0.8 * (S / (self.params['K_s'] + S)) * (self.params['K_i'] / (self.params['K_i'] + P))
            Y_xs_cont = self.params['Y_xs'] * 1.3  # Higher biomass yield
            Y_ps_cont = self.params['Y_ps'] * 0.6  # Lower ethanol yield
            alpha_cont = self.params['alpha'] * 0.3  # Reduced non-growth ethanol production
            
        elif contamination_type == 'wild_yeast':
            # Wild yeast: Different kinetic parameters
            if S < 1e-6:
                mu = 0.0
            else:
                mu = self.params['mu_max'] * 1.2 * (S / (self.params['K_s'] * 1.5 + S)) * (self.params['K_i'] * 0.8 / (self.params['K_i'] * 0.8 + P))
            Y_xs_cont = self.params['Y_xs'] * 0.9
            Y_ps_cont = self.params['Y_ps'] * 0.8
            alpha_cont = self.params['alpha'] * 1.5
            
        elif contamination_type == 'acidic':
            # Acidic contamination: Reduced overall activity
            if S < 1e-6:
                mu = 0.0
            else:
                mu = self.params['mu_max'] * 0.6 * (S / (self.params['K_s'] + S)) * (self.params['K_i'] / (self.params['K_i'] + P))
            Y_xs_cont = self.params['Y_xs'] * 0.7
            Y_ps_cont = self.params['Y_ps'] * 0.7
            alpha_cont = self.params['alpha'] * 0.5
            
        else:
            raise ValueError(f"Unknown contamination type: {contamination_type}")
        
        # Rate equations with contamination effects
        dX_dt = mu * X - self.params['k_d'] * X
        
        # Glucose consumption - stop when glucose is depleted
        if S < 1e-6:
            dS_dt = 0.0
        else:
            dS_dt = -mu * X / Y_xs_cont - alpha_cont * X
            # Ensure glucose consumption doesn't exceed available glucose
            max_consumption = S / (1/60)  # Maximum consumption per hour
            dS_dt = max(dS_dt, -max_consumption)
        
        # Ethanol production
        dP_dt = mu * X / Y_ps_cont + alpha_cont * X
        
        return [dX_dt, dS_dt, dP_dt]
    
    def simulate_episode(self, initial_conditions: List[float], episode_length: int = 48, 
                        contamination_step: Optional[int] = None, 
                        contamination_type: str = 'bacterial') -> pd.DataFrame:
        """
        Simulate a single fermentation episode with fixed length
        
        Args:
            initial_conditions: [X0, S0, P0] - Initial biomass, glucose, ethanol
            episode_length: Length of episode in hours (default: 48 hours)
            contamination_step: Step at which contamination occurs (None for no contamination)
            contamination_type: Type of contamination ('bacterial', 'wild_yeast', 'acidic')
        
        Returns:
            DataFrame with episode simulation results
        """
        
        results = []
        current_state = initial_conditions
        contaminated = False
        
        for step in range(episode_length):
            # Check if contamination starts at this step
            if contamination_step is not None and step == contamination_step:
                contaminated = True
            
            # Choose dynamics based on contamination status
            if contaminated:
                dynamics_func = lambda state, time: self.contaminated_dynamics(state, contamination_type)
            else:
                dynamics_func = self.fermentation_dynamics
            
            # Integrate over 1 hour
            time_span = np.array([0, 1])  # 1 hour
            solution = odeint(dynamics_func, current_state, time_span)
            current_state = solution[-1]
            
            # Apply physical constraints - concentrations cannot be negative
            current_state[0] = max(0.0, current_state[0])  # Biomass >= 0
            current_state[1] = max(0.0, current_state[1])  # Glucose >= 0
            current_state[2] = max(0.0, current_state[2])  # Ethanol >= 0
            
            # Store results
            results.append({
                'step': step,
                'time_hours': step,
                'biomass': current_state[0],
                'glucose': current_state[1],
                'ethanol': current_state[2],
                'contaminated': contaminated,
                'contamination_type': contamination_type if contaminated else 'none'
            })
        
        return pd.DataFrame(results)
    
    def plot_episode(self, data: pd.DataFrame, title: str = "Fermentation Episode"):
        """
        Plot fermentation episode results
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Color contaminated periods
        contaminated_mask = data['contaminated']
        time_hours = data['time_hours']
        
        # Biomass
        ax1.plot(time_hours, data['biomass'], 'b-', linewidth=2, label='Biomass')
        if contaminated_mask.any():
            ax1.fill_between(time_hours, 0, data['biomass'].max()*1.1, 
                           where=contaminated_mask, alpha=0.3, color='red', label='Contaminated')
        ax1.set_xlabel('Time (h)')
        ax1.set_ylabel('Biomass (g/L)')
        ax1.set_title('Biomass Concentration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Glucose
        ax2.plot(time_hours, data['glucose'], 'g-', linewidth=2, label='Glucose')
        if contaminated_mask.any():
            ax2.fill_between(time_hours, 0, data['glucose'].max()*1.1, 
                           where=contaminated_mask, alpha=0.3, color='red', label='Contaminated')
        ax2.set_xlabel('Time (h)')
        ax2.set_ylabel('Glucose (g/L)')
        ax2.set_title('Glucose Concentration')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Ethanol
        ax3.plot(time_hours, data['ethanol'], 'r-', linewidth=2, label='Ethanol')
        if contaminated_mask.any():
            ax3.fill_between(time_hours, 0, data['ethanol'].max()*1.1, 
                           where=contaminated_mask, alpha=0.3, color='red', label='Contaminated')
        ax3.set_xlabel('Time (h)')
        ax3.set_ylabel('Ethanol (g/L)')
        ax3.set_title('Ethanol Concentration')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # All together
        ax4.plot(time_hours, data['biomass'], 'b-', linewidth=2, label='Biomass')
        ax4.plot(time_hours, data['glucose'], 'g-', linewidth=2, label='Glucose')
        ax4.plot(time_hours, data['ethanol'], 'r-', linewidth=2, label='Ethanol')
        if contaminated_mask.any():
            ax4.fill_between(time_hours, 0, max(data['biomass'].max(), data['glucose'].max(), data['ethanol'].max())*1.1, 
                           where=contaminated_mask, alpha=0.3, color='red', label='Contaminated')
        ax4.set_xlabel('Time (h)')
        ax4.set_ylabel('Concentration (g/L)')
        ax4.set_title('Overall Fermentation Profile')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return fig 