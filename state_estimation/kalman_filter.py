import numpy as np
from scipy.stats import chi2

class LinearKalmanFilter:
    """
    Generalized discrete-time Linear Kalman Filter.
    
    Operates on the state transition model:
        x_k = Phi * x_{k-1} + w_k,   w_k ~ N(0, Q)
        z_k = H * x_k + v_k,         v_k ~ N(0, R)
    """
    
    def __init__(self, Phi: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, x0: np.ndarray, P0: np.ndarray = None):
        """
        Initializes the linear Kalman filter matrices.
        
        Args:
            Phi (np.ndarray): State transition matrix (n x n).
            Q (np.ndarray): Process noise covariance matrix (n x n).
            H (np.ndarray): Measurement matrix (m x n).
            R (np.ndarray): Measurement noise covariance matrix (m x m).
            x0 (np.ndarray): Initial state estimate (n x 1) or (n,).
            P0 (np.ndarray, optional): Initial state covariance (n x n). Defaults to identity if not provided.
        """
        self.n = Phi.shape[0]
        self.m = H.shape[0]
        
        # System matrices
        self.Phi = Phi
        self.Q = Q
        self.H = H
        self.R = R
        
        # Current state and covariance
        # Ensure x is a 1D vector mathematically for internal use, or keep it consistent
        self.x_plus = np.asarray(x0, dtype=float).flatten()
        self.P_plus = np.asarray(P0, dtype=float) if P0 is not None else np.eye(self.n)
        
        # Intermediate prediction state
        self.x_minus = np.zeros_like(self.x_plus)
        self.P_minus = np.zeros_like(self.P_plus)
        
        # Innovation and its covariance
        self.v = np.zeros(self.m)
        self.S = np.zeros((self.m, self.m))
        self.K = np.zeros((self.n, self.m))
        
        self.test_innovation_val = 0.0

    def predict(self) -> np.ndarray:
        """
        Time Update (Prediction Step).
        
        x_{k}^{-} = Phi * x_{k-1}^{+}
        P_{k}^{-} = Phi * P_{k-1}^{+} * Phi^T + Q
        """
        self.x_minus = self.Phi @ self.x_plus
        self.P_minus = self.Phi @ self.P_plus @ self.Phi.T + self.Q
        return self.x_minus.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Measurement Update (Correction Step).
        
        v_k = z_k - H * x_{k}^{-}
        S_k = H * P_{k}^{-} * H^T + R
        K_k = P_{k}^{-} * H^T * S_k^{-1}
        x_{k}^{+} = x_{k}^{-} + K_k * v_k
        P_{k}^{+} = (I - K_k * H) * P_{k}^{-}
        
        Args:
            z (np.ndarray): The measurement vector (m x 1) or (m,).
            
        Returns:
            np.ndarray: The updated state vector x_plus.
        """
        z = np.asarray(z, dtype=float).flatten()
        
        # Innovation
        self.v = z - self.H @ self.x_minus
        
        # Innovation covariance
        self.S = self.H @ self.P_minus @ self.H.T + self.R
        
        # Kalman Gain
        self.K = self.P_minus @ self.H.T @ np.linalg.inv(self.S)
        
        # State update
        self.x_plus = self.x_minus + self.K @ self.v
        
        # Covariance update (Joseph form for numeric stability)
        I = np.eye(self.n)
        term1 = I - self.K @ self.H
        self.P_plus = term1 @ self.P_minus @ term1.T + self.K @ self.R @ self.K.T
        
        # Calculate innovation test statistic
        self.test_innovation_val = self.v.T @ np.linalg.inv(self.S) @ self.v
        
        return self.x_plus.copy()

    def check_innovation(self, alpha: float = 0.05) -> bool:
        """
        Performs a Chi-Square test on the innovation to detect anomalies.
        
        Args:
            alpha (float): Significance level (e.g. 0.05 for 95% confidence).
            
        Returns:
            bool: True if observation is rejected (anomaly), False if accepted.
        """
        # Degrees of freedom = number of measurements (m)
        threshold = chi2.ppf(1 - alpha, df=self.m)
        return self.test_innovation_val > threshold
