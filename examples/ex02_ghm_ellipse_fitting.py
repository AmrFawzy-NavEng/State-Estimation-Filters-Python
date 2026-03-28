"""
Example 02: Ellipse Fitting with Gauss-Helmert Model
======================================================
Demonstrates the `GaussHelmertEstimator` solving a rigorous
implicit nonlinear least squares problem: Fitting an ellipse
to 2D observation coordinates commonly found in photogrammetry.

The implicit ellipse model is:
f(x, l) = (x_bar^2 / a^2) + (y_bar^2 / b^2) - 1 = 0

Where (x_bar, y_bar) are the rotated/translated coordinates:
x_bar = (x_i - x_M)*cos(t) + (y_i - y_M)*sin(t)
y_bar = -(x_i - x_M)*sin(t) + (y_i - y_M)*cos(t)

Parameters (x): [x_M, y_M, a, b, t] (Center, Semi-axes, Rotation)
Observations (l): [x_1, y_1, x_2, y_2, ...]
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from state_estimation import GaussHelmertEstimator

# ----------------------------------------------------
# 1. Functional Model Callables for the GHM Estimator
# ----------------------------------------------------

def ellipse_w(x: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Computes the misclosure vector w = f(x, l)."""
    xM, yM, a, b, t = x
    
    pts = l.reshape(-1, 2)
    xi, yi = pts[:, 0], pts[:, 1]
    
    ct, st = np.cos(t), np.sin(t)
    dx, dy = xi - xM, yi - yM
    
    x_bar = dx * ct + dy * st
    y_bar = -dx * st + dy * ct
    
    return (x_bar**2 / a**2) + (y_bar**2 / b**2) - 1

def ellipse_A(x: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Computes the Design Matrix A (Jacobian w.r.t parameters x)."""
    xM, yM, a, b, t = x
    pts = l.reshape(-1, 2)
    xi, yi = pts[:, 0], pts[:, 1]
    n = len(xi)
    
    ct, st = np.cos(t), np.sin(t)
    dx, dy = xi - xM, yi - yM
    x_bar = dx * ct + dy * st
    y_bar = -dx * st + dy * ct
    
    A = np.zeros((n, 5))
    
    # df/dxM, df/dyM
    A[:, 0] = (2 * x_bar / a**2) * (-ct) + (2 * y_bar / b**2) * (st)
    A[:, 1] = (2 * x_bar / a**2) * (-st) + (2 * y_bar / b**2) * (-ct)
    
    # df/da, df/db
    A[:, 2] = -2 * x_bar**2 / a**3
    A[:, 3] = -2 * y_bar**2 / b**3
    
    # df/dt (rotation)
    A[:, 4] = 2 * x_bar * y_bar * (1/a**2 - 1/b**2)
    
    return A

def ellipse_B(x: np.ndarray, l: np.ndarray) -> np.ndarray:
    """Computes the Condition Matrix B (Jacobian w.r.t observations l)."""
    xM, yM, a, b, t = x
    pts = l.reshape(-1, 2)
    xi, yi = pts[:, 0], pts[:, 1]
    n = len(xi)
    
    ct, st = np.cos(t), np.sin(t)
    dx, dy = xi - xM, yi - yM
    x_bar = dx * ct + dy * st
    y_bar = -dx * st + dy * ct
    
    B = np.zeros((n, 2 * n))
    
    # df/dxi, df/dyi
    df_dxi = (2 * x_bar / a**2) * (ct) + (2 * y_bar / b**2) * (-st)
    df_dyi = (2 * x_bar / a**2) * (st) + (2 * y_bar / b**2) * (ct)
    
    for i in range(n):
        B[i, 2*i]     = df_dxi[i]
        B[i, 2*i + 1] = df_dyi[i]
        
    return B

# ----------------------------------------------------
# 2. Main Execution
# ----------------------------------------------------

def main():
    print("--- Implicit Non-Linear Least Squares: Ellipse Fitting (GHM) ---")
    
    # Initial Approximate values: center=(25,20), a=4.5, b=2.5, rot=20g
    x0_start = np.array([25.0, 20.0, 4.5, 2.5, 20 * np.pi / 200])
    
    # 8 Observed points on the ellipse boundary
    observations = np.array([
        [24.501, 24.046],
        [27.227, 24.089],
        [29.395, 22.091],
        [28.040, 20.022],
        [25.243, 18.253],
        [21.662, 18.581],
        [20.906, 21.292],
        [22.461, 22.790]
    ])
    
    # Flatten properly for the general solver (C-style order: x1, y1, x2, y2...)
    l_raw = observations.flatten()
    
    # Uncorrelated homoscedastic coordinates (s_0 = 0.01^2 a priori weight implies Q=I functionally)
    Q_ll = np.eye(len(l_raw))
    
    # Initialize robust estimator
    estimator = GaussHelmertEstimator(
        func_W=ellipse_w, 
        func_A=ellipse_A, 
        func_B=ellipse_B
    )
    
    print("Run iterative solver...")
    success = estimator.fit(l=l_raw, x0=x0_start, Q_ll=Q_ll, tol=1e-5)
    
    if not success:
        print("GHM solver failed to converge.")
        return
        
    x_hat = estimator.x_hat
    std_devs = estimator.get_standard_deviations()
    
    print("\\nConvergence Successful!")
    print(f"Aposteriori Variance Factor (sigma0^2): {estimator.sigma02_hat:.6f}")
    
    print("\\nFinal Adjusted Parameters:")
    print(f"X Center = {x_hat[0]:.4f} m  (Std: {std_devs[0]:.4f} m)")
    print(f"Y Center = {x_hat[1]:.4f} m  (Std: {std_devs[1]:.4f} m)")
    print(f"Semi-Major (a) = {x_hat[2]:.4f} m  (Std: {std_devs[2]:.4f} m)")
    print(f"Semi-Minor (b) = {x_hat[3]:.4f} m  (Std: {std_devs[3]:.4f} m)")
    print(f"Rotation (t)   = {np.rad2deg(x_hat[4]):.4f}°  (Std: {np.rad2deg(std_devs[4]):.4f}°)")
    
    # Visualization
    plt.figure(figsize=(8, 8))
    
    # Raw points
    plt.plot(observations[:,0], observations[:,1], 'ro', label='Raw Observations', markersize=6)
    
    # Adjusted points
    l_adj = estimator.l_hat.reshape(-1, 2)
    plt.plot(l_adj[:,0], l_adj[:,1], 'b+', label='Adjusted Observations', markersize=8)
    
    # Draw exact analytic ellipse contour
    angles = np.linspace(0, 2*np.pi, 200)
    x_ellipse = x_hat[0] + x_hat[2]*np.cos(angles)*np.cos(x_hat[4]) - x_hat[3]*np.sin(angles)*np.sin(x_hat[4])
    y_ellipse = x_hat[1] + x_hat[2]*np.cos(angles)*np.sin(x_hat[4]) + x_hat[3]*np.sin(angles)*np.cos(x_hat[4])
    
    plt.plot(x_ellipse, y_ellipse, 'k--', label='Fitted GHM Ellipse')
    plt.plot(x_hat[0], x_hat[1], 'gx', label='Center', markersize=10)
    
    plt.title('Implicit Ellipse Fitting via Gauss-Helmert Model')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    out_path = os.path.join(os.path.dirname(__file__), 'ellipse_ghm_fit.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\\nSaved plot to '{out_path}'.")

if __name__ == "__main__":
    main()
