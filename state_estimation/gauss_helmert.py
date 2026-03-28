import numpy as np

class GaussHelmertEstimator:
    """
    Generalized Gauss-Helmert Model (Implicit Non-Linear Least Squares Estimator).
    
    Solves the implicit condition equations:
        f(l_hat, x_hat) = 0
    where 'l_hat' are the adjusted observations (l + v) and 'x_hat' are the adjusted parameters (x0 + dx).
    """

    def __init__(self, func_W, func_A, func_B):
        """
        Initializes the GHM estimator with the required functional model components.
        
        Args:
            func_W (callable): Returns misclosure vector w = f(x, l). Signature: f(x, l) -> np.ndarray
            func_A (callable): Returns Design matrix A = df/dx(x, l). Signature: f(x, l) -> np.ndarray
            func_B (callable): Returns Condition matrix B = df/dl(x, l). Signature: f(x, l) -> np.ndarray
        """
        self.func_W = func_W
        self.func_A = func_A
        self.func_B = func_B
        
        # Outputs
        self.x_hat = None
        self.l_hat = None
        self.v = None
        self.sigma02_hat = None
        self.Sigma_xx = None

    def fit(self, l: np.ndarray, x0: np.ndarray, Q_ll: np.ndarray = None, max_iter: int = 100, tol: float = 1e-10) -> bool:
        """
        Iteratively estimates the parameters using the Gauss-Helmert algorithm.
        
        Args:
            l (np.ndarray): The raw observation vector.
            x0 (np.ndarray): The prior approximate values for the parameters.
            Q_ll (np.ndarray, optional): Observation cofactor matrix. Defaults to Identity.
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence criteria for max(abs(dx)).
            
        Returns:
            bool: True if converged successfully, False otherwise.
        """
        n_obs = len(l)
        if Q_ll is None:
            Q_ll = np.eye(n_obs)
            
        # Initialize
        x_k = x0.copy()
        l_obj = l.copy()
        l_k = l_obj.copy()  # Linearization point for observations
        
        for iteration in range(max_iter):
            # Compute matrices at current linearization point (x_k, l_k)
            A = self.func_A(x_k, l_k)
            B = self.func_B(x_k, l_k)
            w_val = self.func_W(x_k, l_k)
            
            # The true misclosure includes the Taylor expansion offset for observations
            w = w_val + B @ (l_obj - l_k)
            
            # Sub-matrices
            Q_ww = B @ Q_ll @ B.T
            Q_ww_inv = np.linalg.inv(Q_ww)
            
            # Normal equations
            N = A.T @ Q_ww_inv @ A
            n_vec = A.T @ Q_ww_inv @ (-w)
            
            # Solve for parameter update
            dx = np.linalg.solve(N, n_vec)
            x_k = x_k + dx
            
            # Solve for correlates
            k = Q_ww_inv @ (-w - A @ dx)
            
            # Compute observation residuals
            v = Q_ll @ B.T @ k
            
            # Update observation linearization point
            l_k = l_obj + v
            
            # Check convergence
            if np.max(np.abs(dx)) < tol:
                self.x_hat = x_k
                self.l_hat = l_k
                self.v = v
                
                # Degrees of freedom (r) = Number of conditions (c) - Number of unknowns (u)
                num_conditions = w.shape[0]
                num_unknowns = x0.shape[0]
                r = num_conditions - num_unknowns
                
                # Aposteriori variance factor
                self.sigma02_hat = (v.T @ v) / r if r > 0 else 0.0
                
                # Final covariance matrices
                A_final = self.func_A(self.x_hat, self.l_hat)
                B_final = self.func_B(self.x_hat, self.l_hat)
                Q_ww_final = B_final @ Q_ll @ B_final.T
                N_final = A_final.T @ np.linalg.inv(Q_ww_final) @ A_final
                
                Q_xx = np.linalg.inv(N_final)
                self.Sigma_xx = self.sigma02_hat * Q_xx
                
                return True
                
        return False
        
    def get_standard_deviations(self) -> np.ndarray:
        """Returns the standard deviations of the estimated parameters."""
        if self.Sigma_xx is not None:
            return np.sqrt(np.diag(self.Sigma_xx))
        return None
