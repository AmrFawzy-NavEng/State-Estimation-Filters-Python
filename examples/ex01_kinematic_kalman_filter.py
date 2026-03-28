"""
Example 01: Kinematic Tracking with Linear Kalman Filter
========================================================
Demonstrates how to use the `LinearKalmanFilter` to track a
2D platform undergoing uniformly accelerated motion (constant acceleration model).

The state vector is x = [px, py, vx, vy, ax, ay]^T.
We generate a synthetic true trajectory, add process and measurement noise,
and then apply the Kalman filter to estimate the optimal states.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import our package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from state_estimation import LinearKalmanFilter

def create_system_matrices(dt: float, q_noise: float, r_pos_noise: float):
    """
    Creates the Phi, Q, H, R matrices for a 2D uniformly accelerated model.
    """
    # 1. State Transition Matrix (Phi)
    # State: [px, py, vx, vy, ax, ay]
    Phi = np.eye(6)
    
    # Velocity contribution to position
    Phi[0, 2] = dt
    Phi[1, 3] = dt
    
    # Acceleration contribution to position
    Phi[0, 4] = 0.5 * dt**2
    Phi[1, 5] = 0.5 * dt**2
    
    # Acceleration contribution to velocity
    Phi[2, 4] = dt
    Phi[3, 5] = dt
    
    # 2. Process Noise Covariance (Q) - Discrete Constant Acceleration Model (Bar-Shalom)
    q11 = q_noise * (dt**5) / 20.0
    q12 = q_noise * (dt**4) / 8.0
    q13 = q_noise * (dt**3) / 6.0
    q22 = q_noise * (dt**3) / 3.0
    q23 = q_noise * (dt**2) / 2.0
    q33 = q_noise * dt

    # Q for a single dimension
    Q_1d = np.array([
        [q11, q12, q13],
        [q12, q22, q23],
        [q13, q23, q33]
    ])
    
    Q = np.zeros((6, 6))
    Q[0::2, 0::2] = Q_1d # x-coordinates
    Q[1::2, 1::2] = Q_1d # y-coordinates
    
    # 3. Measurement Matrix (H)
    # Assume we only measure Positions (px, py)
    H = np.zeros((2, 6))
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    
    # 4. Measurement Noise Covariance (R)
    R = np.eye(2) * (r_pos_noise**2)
    
    return Phi, Q, H, R

def generate_trajectory(n_epochs: int, dt: float, Phi: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray):
    """Generates ground truth and noisy measurements."""
    np.random.seed(42)
    
    x_true = np.zeros((6, n_epochs))
    z_meas = np.zeros((2, n_epochs))
    
    # Initial state (start moving with some acceleration)
    x_true[:, 0] = [0.0, 0.0, 10.0, 5.0, 0.5, -0.2]
    
    # Generate true states and process/measurement noise
    for k in range(1, n_epochs):
        # True dynamics + Process Noise
        w_k = np.random.multivariate_normal(np.zeros(6), Q)
        x_true[:, k] = Phi @ x_true[:, k-1] + w_k
        
        # Add a sudden maneuver anomaly at epoch 40
        if k == 40:
            x_true[4:6, k] += [2.0, -1.0] # Sudden acceleration spike
            
    for k in range(n_epochs):
        # Measurements + Measurement Noise
        v_k = np.random.multivariate_normal(np.zeros(2), R)
        z_meas[:, k] = H @ x_true[:, k] + v_k
        
    return x_true, z_meas

def main():
    print("--- Kinematic 2D Tracking with Kalman Filter ---")
    
    dt = 0.5
    n_epochs = 60
    q_noise = 0.2  # Process noise spectral density
    r_pos = 5.0    # Measurement noise standard deviation (meters)
    
    Phi, Q, H, R = create_system_matrices(dt, q_noise, r_pos)
    
    print("Generating synthetic trajectory and noisy sensor data...")
    x_true, z_meas = generate_trajectory(n_epochs, dt, Phi, Q, H, R)
    
    print("Initializing Linear Kalman Filter...")
    # Initial guess with high uncertainty
    x0 = np.zeros(6)
    P0 = np.eye(6) * 100.0
    kf = LinearKalmanFilter(Phi, Q, H, R, x0, P0)
    
    x_est = np.zeros((6, n_epochs))
    anomalies = []
    
    for k in range(n_epochs):
        # 1. Prediction step
        kf.predict()
        
        # 2. Update step
        x_est[:, k] = kf.update(z_meas[:, k])
        
        # 3. Innovation Test (Anomaly detection)
        # Using 99% confidence (alpha=0.01)
        is_anomaly = kf.check_innovation(alpha=0.01)
        if is_anomaly:
            anomalies.append(k)
            print(f"  Anomaly detected at epoch {k}! Statistical Innovation rejected.")
            
    print(f"\\nFiltering complete. Processed {n_epochs} epochs.")
    
    # Error analysis
    rmse = np.sqrt(np.mean((x_true[0:2, :] - x_est[0:2, :])**2, axis=1))
    print(f"Position RMSE (X, Y): [{rmse[0]:.2f}m, {rmse[1]:.2f}m]")
    
    # Visualizations
    plt.figure(figsize=(10, 6))
    
    # Plot Trajectories
    plt.plot(x_true[0, :], x_true[1, :], 'k-', linewidth=2, label='True Trajectory')
    plt.plot(z_meas[0, :], z_meas[1, :], 'rx', markersize=6, alpha=0.5, label='Noisy GPS Measurements')
    plt.plot(x_est[0, :], x_est[1, :], 'b-', linewidth=2, label='Filter Estimate')
    
    # Highlight anomalies
    for anomaly_idx in anomalies:
        plt.plot(z_meas[0, anomaly_idx], z_meas[1, anomaly_idx], 'yo', markersize=10, 
                 markeredgecolor='k', label='Innovation Anomaly' if anomaly_idx == anomalies[0] else "")
                 
    plt.title('2D Kinematic Kalman Filter Tracking')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    out_path = os.path.join(os.path.dirname(__file__), 'kalman_tracking.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\\nSaved trajectory plot to '{out_path}'")

if __name__ == "__main__":
    main()
