import numpy as np

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        """
        min_cutoff: Minimum cutoff frequency (lower = smoother when still)
        beta: Speed coefficient (higher = less lag during fast movements)
        d_cutoff: Cutoff frequency for derivative
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0.0:
            t_e = 1e-4  # Prevent division by zero

        # Filter the derivative (speed of movement)
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Dynamically adjust cutoff frequency based on movement speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        
        # Filter the actual coordinate
        x_hat = a * x + (1 - a) * self.x_prev

        # Save state for next frame
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat