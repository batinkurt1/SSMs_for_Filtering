import numpy as np

def generate_input_sequence(n_steps, m, umin, umax, seed=None):
    """
    Returns:
      (n_steps, m) array of control inputs
    """
    rng = np.random.default_rng(seed) 
    return rng.uniform(umin, umax, size=(n_steps, m))