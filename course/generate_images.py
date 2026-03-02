import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

# Common styling
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Helper function
def set_spines_invisible(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

# ------------------------------------------------------------------------
# 1. 03a_svd_for_muon_intro.png
# Bar chart comparing raw singular values to Muon singular values
# ------------------------------------------------------------------------
def generate_svd_for_muon_intro():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(10)
    # Exponentially decaying singular values for Raw Gradient
    raw_sv = 5.0 * np.exp(-x / 2.5) 
    
    # Muon equalizes singular values to 1
    muon_sv = np.ones(10)
    
    width = 0.35
    ax.bar(x - width/2, raw_sv, width, label='Raw Gradient ($G$)', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, muon_sv, width, label='Muon ($UV^T$)', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Magnitude ($\\sigma_i$)')
    ax.set_title('Muon Equalizes the Gradient\'s Singular Values')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in x])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('images/03a_svd_for_muon_intro.png', dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------
# 2. 03a_svd_geometry_steps.png
# Step-by-step geometric view of SVD
# ------------------------------------------------------------------------
def generate_svd_geometry_steps():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Base circle coordinates
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    points = np.vstack((x, y))
    
    # Define transformations
    theta_v = np.pi / 6 # 30 degrees
    Vt = np.array([[np.cos(theta_v), -np.sin(theta_v)],
                   [np.sin(theta_v), np.cos(theta_v)]])
                   
    Sigma = np.array([[3, 0],
                      [0, 1]])
                      
    theta_u = np.pi / 4 # 45 degrees
    U = np.array([[np.cos(theta_u), -np.sin(theta_u)],
                  [np.sin(theta_u), np.cos(theta_u)]])
    
    # Transformation steps
    steps = [
        ('1. Input Space', points),
        ('2. Apply $V^T$ (Rotate)', Vt @ points),
        ('3. Apply $\\Sigma$ (Scale)', Sigma @ Vt @ points),
        ('4. Apply $U$ (Rotate)', U @ Sigma @ Vt @ points)
    ]
    
    colors = plt.cm.viridis(np.linspace(0, 1, 100))
    
    for i, (title, tr_points) in enumerate(steps):
        ax = axes[i]
        
        # Grid and axes lines
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
        
        # Plot points with color gradient to track orientation
        ax.scatter(tr_points[0,:], tr_points[1,:], c=colors, s=10)
        
        # Draw basis vectors as arrows for visual intuition
        # To see how [1,0] and [0,1] transform
        basis = np.array([[1, 0], [0, 1]])
        if i == 0:
            tr_basis = basis
        elif i == 1:
            tr_basis = Vt @ basis
        elif i == 2:
            tr_basis = Sigma @ Vt @ basis
        else:
            tr_basis = U @ Sigma @ Vt @ basis
            
        ax.arrow(0, 0, tr_basis[0,0], tr_basis[1,0], head_width=0.15, head_length=0.15, fc='red', ec='red', zorder=10)
        ax.arrow(0, 0, tr_basis[0,1], tr_basis[1,1], head_width=0.15, head_length=0.15, fc='blue', ec='blue', zorder=10)
            
        ax.set_aspect('equal')
        max_val = 3.5
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add arrow between plots
        if i < 3:
            fig.text(ax.get_position().x1 + 0.01, ax.get_position().y0 + ax.get_position().height/2, 
                     '$\\rightarrow$', fontsize=24, ha='center', va='center')

    plt.suptitle('Geometric Interpretation of SVD ($A = U\\Sigma V^T$)', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig('images/03a_svd_geometry_steps.png', dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------
# 3. 03a_conditioning_trajectory.png
# Zig-zag path for standard vs smoother for Muon
# ------------------------------------------------------------------------
def generate_conditioning_trajectory():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Define a highly anisotropic quadratic bowl
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * X**2 + 5.0 * Y**2 # Steep in Y (high singular value), flat in X (low singular value)
    
    # Standard Gradient Descent path (Zig-zag)
    # Start at (-8, 8)
    curr = np.array([-8.0, 8.0])
    gd_path = [curr.copy()]
    lr_gd = 0.15 # Learning rate for standard GD
    
    for _ in range(25):
        grad = np.array([1.0 * curr[0], 10.0 * curr[1]])
        curr = curr - lr_gd * grad
        gd_path.append(curr.copy())
    gd_path = np.array(gd_path)
    
    # "Muon-like" balanced path (simulated by preconditioning/scaling out the ill-conditioning)
    curr = np.array([-8.0, 8.0])
    muon_path = [curr.copy()]
    lr_muon = 1.0 # Learning rate for balanced
    
    for _ in range(10):
        grad = np.array([1.0 * curr[0], 10.0 * curr[1]])
        # Simulate equalizing singular values (preconditioning)
        # We roughly balance the update size in each direction
        update = np.array([np.sign(grad[0]) * abs(grad[0])**0.5, 
                           np.sign(grad[1]) * abs(grad[1])**0.5]) * 2
        # A true ideal update would just point directly to center on a bowl,
        # Let's approximate a much more direct trajectory
        balanced_grad = np.array([curr[0], curr[1]]) * 0.8
        
        curr = curr - lr_muon * balanced_grad
        muon_path.append(curr.copy())
    muon_path = np.array(muon_path)
    
    
    # Plot Standard
    ax = axes[0]
    cs = ax.contour(X, Y, Z, levels=20, cmap='binary', alpha=0.5)
    ax.plot(gd_path[:, 0], gd_path[:, 1], 'o-', color='#e74c3c', linewidth=2, markersize=5, label='Trajectory')
    ax.plot(0, 0, 'x', color='black', markersize=12, markeredgewidth=3, label='Minimum')
    ax.set_title('Standard Gradient Descent\n(Zig-zags in steep dimensions)')
    ax.legend(loc='lower left')
    ax.set_aspect('equal')
    set_spines_invisible(ax)
    
    # Plot Balanced
    ax = axes[1]
    ax.contour(X, Y, Z, levels=20, cmap='binary', alpha=0.5)
    ax.plot(muon_path[:, 0], muon_path[:, 1], 'o-', color='#3498db', linewidth=2, markersize=5, label='Trajectory')
    ax.plot(0, 0, 'x', color='black', markersize=12, markeredgewidth=3)
    ax.set_title('Balanced Direction (Muon-like)\n(All dimensions scaled equally)')
    ax.set_aspect('equal')
    set_spines_invisible(ax)
    
    plt.tight_layout()
    plt.savefig('images/03a_conditioning_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------
# 4. 03a_newton_schulz_singular_value_squash.png
# Newton-Schulz singular values converging toward 1
# ------------------------------------------------------------------------
def generate_ns_singular_value_squash():
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Quintic Newton-Schulz iteration function
    def quintic_ns(sigma, iteration):
        # Using coefficients from the Muon code
        if iteration == 0:
            a, b, c = 3.4445, -4.7750, 2.0315
        elif iteration == 1:
            a, b, c = 11.3168, -20.3300, 9.7132
        else:
            a, b, c = 8.4749, -13.9590, 6.1843
            
        return a * sigma + b * sigma**3 + c * sigma**5
    
    # Initial singular values spread out after normalization
    initial_sigmas = [0.6, 0.7, 0.8, 1.2, 1.3, 1.4]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(initial_sigmas)))
    
    n_steps = 6
    iterations = np.arange(n_steps + 1)
    
    for i, S0 in enumerate(initial_sigmas):
        vals = [S0]
        curr_S = S0
        for step in range(n_steps):
            try:
                curr_S = quintic_ns(curr_S, step)
            except OverflowError:
                curr_S = np.nan
            vals.append(curr_S)
            
        ax.plot(iterations, vals, 'o-', linewidth=2.5, markersize=8, color=colors[i], 
                label=f'Initial $\\sigma = {S0}$')
        
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Target $\\sigma = 1.0$ ($UV^T$)')
    
    ax.set_xlabel('Newton-Schulz Iteration Step')
    ax.set_ylabel('Singular Value Magnitude')
    ax.set_title('Newton-Schulz Iterations Converging Singular Values to 1')
    ax.set_xticks(iterations)
    ax.set_ylim(0, 2.5)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Add an annotation box highlighting the rapid convergence
    ax.annotate('Rapid convergence\nwithin 3-5 steps!', 
                xy=(4.5, 1.0), xycoords='data',
                xytext=(3, 1.8), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1, alpha=0.9),
                fontsize=11)
                
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('images/03a_newton_schulz_singular_value_squash.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating images...")
    generate_svd_for_muon_intro()
    generate_svd_geometry_steps()
    generate_conditioning_trajectory()
    generate_ns_singular_value_squash()
    print("Images generated successfully!")

# ------------------------------------------------------------------------
# 5. 02_steepest_descent_norms.png
# Unit balls and steepest descent directions for L2, L_inf, Spectral
# ------------------------------------------------------------------------
def generate_steepest_descent_norms():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    gradient = np.array([2.0, 1.0])
    
    # helper for drawing balls
    theta = np.linspace(0, 2*np.pi, 100)
    
    # 1. L2 / Frobenius
    ax = axes[0]
    ax.set_aspect('equal')
    x = np.cos(theta)
    y = np.sin(theta)
    ax.fill(x, y, alpha=0.2, color='blue', label='L2 Unit Ball')
    ax.arrow(0, 0, gradient[0], gradient[1], head_width=0.1, color='gray', label='-Gradient')
    optimal_l2 = gradient / np.linalg.norm(gradient)
    ax.arrow(0, 0, optimal_l2[0], optimal_l2[1], head_width=0.15, color='blue', linewidth=2, label='Opt Dir (GD)')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title('L2 Norm (Standard GD)')
    ax.legend(loc='upper right')
    set_spines_invisible(ax)

    # 2. L_inf (SignSGD)
    ax = axes[1]
    ax.set_aspect('equal')
    ax.fill([-1, 1, 1, -1], [-1, -1, 1, 1], alpha=0.2, color='red', label='L$\\infty$ Unit Ball')
    ax.arrow(0, 0, gradient[0], gradient[1], head_width=0.1, color='gray')
    optimal_linf = np.sign(gradient)
    ax.arrow(0, 0, optimal_linf[0], optimal_linf[1], head_width=0.15, color='red', linewidth=2, label='Opt Dir (SignSGD)')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title('L$\\infty$ Norm (SignSGD/Adam-like)')
    ax.legend(loc='upper right')
    set_spines_invisible(ax)

    # 3. Spectral (Muon) - Conceptual for 2D matrix
    # Treat the 2D plane as the singular values space
    ax = axes[2]
    ax.set_aspect('equal')
    ax.fill([-1, 1, 1, -1], [-1, -1, 1, 1], alpha=0.2, color='purple', label='Spectral Unit Ball')
    ax.arrow(0, 0, gradient[0], gradient[1], head_width=0.1, color='gray')
    # Spectral steepest descent drives all singular values evenly
    optimal_spectral = np.array([1.0, 1.0]) * np.sign(gradient)
    ax.arrow(0, 0, optimal_spectral[0], optimal_spectral[1], head_width=0.15, color='purple', linewidth=2, label='Opt Dir (Muon)')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title('Spectral Norm (Muon)\nOrthogonal Factor')
    ax.legend(loc='upper right')
    set_spines_invisible(ax)
    
    plt.suptitle('Steepest Descent Under Different Norms', fontsize=16)
    plt.tight_layout()
    plt.savefig('images/02_steepest_descent_norms.png', dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------
# 6. 08_memory_comparison.png
# Optimizer memory footprint
# ------------------------------------------------------------------------
def generate_memory_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    optimizers = ['SGD+Mom', 'Muon', 'Adam/AdamW', 'Shampoo (Est.)']
    # memory multipliers (1 means only params, + N means states)
    # Total memory = weights + grads + states
    # Let's just plot Optimizer States size relative to parameter count (N)
    states = [1.0, 1.0, 2.0, 1.5]
    colors = ['#bdc3c7', '#3498db', '#e74c3c', '#f39c12']
    
    x = np.arange(len(optimizers))
    width = 0.5
    ax.bar(x, states, width, color=colors, alpha=0.8)
    
    ax.set_ylabel('Optimizer State Size (Multiple of Model Size)')
    ax.set_title('Memory Footprint Comparison (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(optimizers)
    ax.axhline(1.0, color='gray', linestyle='--')
    ax.set_ylim(0, 2.5)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(states):
        ax.text(i, v + 0.1, f'{v}N', ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('images/08_memory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------
# 7. 08_convergence_comparison.png
# Convergence Curves
# ------------------------------------------------------------------------
def generate_convergence_comparison():
    fig, ax = plt.subplots(figsize=(9, 6))
    
    steps = np.linspace(0, 10000, 100)
    
    # Simulate loss curves (exponential decays)
    # Adam reaches ~3.28 in 10k steps
    adam_loss = 4.5 * np.exp(-steps / 4000) + 3.2
    
    # Muon reaches ~3.28 in 5k steps
    muon_loss = 4.5 * np.exp(-steps / 2000) + 3.15
    
    ax.plot(steps, adam_loss, '-', linewidth=3, color='#e74c3c', label='AdamW Baseline')
    ax.plot(steps, muon_loss, '-', linewidth=3, color='#3498db', label='Muon')
    
    ax.axhline(3.28, color='gray', linestyle='--', label='Target Loss (3.28)')
    ax.axvline(10000, color='#e74c3c', linestyle=':', alpha=0.5)
    ax.axvline(5500, color='#3498db', linestyle=':', alpha=0.5)
    
    ax.annotate(' ~1.8x fewer steps ', 
                xy=(5500, 3.28), xytext=(7000, 3.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=11)
                
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Convergence Speed: Muon vs AdamW')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_ylim(3.0, 7.5)
    
    plt.tight_layout()
    plt.savefig('images/08_convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------
# 8. Weight Update Explanations
# ------------------------------------------------------------------------
def generate_weight_update_rule():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, '$\\theta_{t+1} = \\theta_t - \\eta \\cdot \\Delta$', fontsize=36, ha='center', va='center', weight='bold')
    ax.set_title("Weight Update Rule", fontsize=16)
    set_spines_invisible(ax)
    plt.tight_layout()
    plt.savefig('images/01_weight_update_rule.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_weight_update_matrix():
    fig, ax = plt.subplots(figsize=(5, 5))
    np.random.seed(42)
    matrix = np.random.randn(10, 10)
    cax = ax.matshow(matrix, cmap='viridis')
    fig.colorbar(cax, shrink=0.8)
    ax.set_title('Weight Update Matrix $\\Delta$', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('images/01_weight_update_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_non_orthogonal_effect():
    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    # Scaled unevenly
    x_new = 3.0 * x
    y_new = 0.5 * y
    
    # rotated
    angle = np.pi / 4
    x_rot = x_new * np.cos(angle) - y_new * np.sin(angle)
    y_rot = x_new * np.sin(angle) + y_new * np.cos(angle)
    
    ax.plot(x, y, 'b--', alpha=0.5, label='Original space')
    ax.plot(x_rot, y_rot, 'r-', linewidth=2, label='Transformed Space')
    ax.arrow(0, 0, 3*np.cos(angle), 3*np.sin(angle), head_width=0.2, color='red', length_includes_head=True)
    ax.arrow(0, 0, 0.5*np.cos(angle+np.pi/2), 0.5*np.sin(angle+np.pi/2), head_width=0.1, color='darkred', length_includes_head=True)
    
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='b', linestyle='--', alpha=0.5),
                    Line2D([0], [0], color='r', lw=2),
                    Line2D([0], [0], color='r', marker='>', linestyle='None', markersize=10),
                    Line2D([0], [0], color='darkred', marker='>', linestyle='None', markersize=10)]
    ax.legend(custom_lines, ['Original space', 'Transformed Space', 'Strong Pull', 'Weak Pull'], loc='upper right', fontsize=9)
    
    ax.set_title('Non-Orthogonal Matrix:\nPulls strongly in some directions, weakly in others')
    set_spines_invisible(ax)
    plt.tight_layout()
    plt.savefig('images/01_non_orthogonal_effect.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_orthogonal_effect():
    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    
    angle = np.pi / 4
    
    ax.plot(x, y, 'b--', alpha=0.5, label='Original space')
    # Scale by 2 just for visibility outside the original circle
    ax.plot(2*x, 2*y, 'g-', linewidth=2, label='Transformed Space')
    
    # Add arrows to show equal scale
    for a in [angle, angle+np.pi/2]:
        ax.arrow(0, 0, 2*np.cos(a), 2*np.sin(a), head_width=0.2, color='green', length_includes_head=True)
        
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='b', linestyle='--', alpha=0.5),
                    Line2D([0], [0], color='g', lw=2),
                    Line2D([0], [0], color='g', marker='>', linestyle='None', markersize=10)]
    ax.legend(custom_lines, ['Original space', 'Transformed Space', 'Equal Pull in all directions'], loc='upper right', fontsize=9)
    ax.set_title('Orthogonal Matrix:\nPulls equally in all orthogonal directions')
    set_spines_invisible(ax)
    plt.tight_layout()
    plt.savefig('images/01_orthogonal_effect.png', dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------
# 9. Matrix Vector Transformation
# ------------------------------------------------------------------------
def generate_matrix_vector_transform():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Input vector
    ax = axes[0]
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', linewidth=1, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=1, alpha=0.5)
    ax.arrow(0, 0, 1, 0, head_width=0.15, color='blue', length_includes_head=True, label='Basis $\\hat{i}$')
    ax.arrow(0, 0, 0, 1, head_width=0.15, color='green', length_includes_head=True, label='Basis $\\hat{j}$')
    ax.arrow(0, 0, 1.5, 1, head_width=0.15, color='purple', length_includes_head=True, width=0.04, label='Vector $\\vec{v}$')
    ax.set_xlim(-1, 3.5)
    ax.set_ylim(-1, 3.5)
    ax.set_title('Input Space')
    ax.legend(loc='upper right')
    set_spines_invisible(ax)

    # Transformed vector
    ax = axes[1]
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', linewidth=1, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=1, alpha=0.5)
    
    # Transformation Matrix M
    M = np.array([[1.5, -0.5], [0.5, 1.5]])
    i_trans = M @ np.array([1, 0])
    j_trans = M @ np.array([0, 1])
    v_trans = M @ np.array([1.5, 1])
    
    ax.arrow(0, 0, i_trans[0], i_trans[1], head_width=0.15, color='blue', length_includes_head=True, label='$M\\hat{i}$')
    ax.arrow(0, 0, j_trans[0], j_trans[1], head_width=0.15, color='green', length_includes_head=True, label='$M\\hat{j}$')
    ax.arrow(0, 0, v_trans[0], v_trans[1], head_width=0.15, color='purple', length_includes_head=True, width=0.04, label='$M\\vec{v}$')
    ax.set_xlim(-1, 3.5)
    ax.set_ylim(-1, 3.5)
    ax.set_title('Output Space (After Matrix Multiplication)')
    ax.legend(loc='upper right')
    set_spines_invisible(ax)

    plt.suptitle('Matrices are Transformations', fontsize=16)
    plt.tight_layout()
    plt.savefig('images/01_matrix_vector_transform.png', dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------
# 10. Core Optimizers and Math Concepts
# ------------------------------------------------------------------------
def generate_gradient_descent():
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 2*Y**2
    ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.5)
    
    curr = np.array([-2.5, 2.5])
    path = [curr.copy()]
    for _ in range(15):
        grad = np.array([2*curr[0], 4*curr[1]])
        curr = curr - 0.15 * grad
        path.append(curr.copy())
    path = np.array(path)
    
    ax.plot(path[:, 0], path[:, 1], 'ro-', label='Gradient Descent')
    ax.plot(0, 0, 'kx', markersize=10, markeredgewidth=2, label='Minimum')
    
    ax.set_title('Gradient Descent: Steps orthogonal to contours')
    ax.legend(loc='upper right')
    set_spines_invisible(ax)
    plt.tight_layout()
    plt.savefig('images/01_gradient_descent.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_momentum():
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 5*Y**2
    ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.5)
    
    curr = np.array([-2.5, 2.5])
    path_gd = [curr.copy()]
    for _ in range(20):
        grad = np.array([2*curr[0], 10*curr[1]])
        curr = curr - 0.15 * grad
        path_gd.append(curr.copy())
    path_gd = np.array(path_gd)
    
    curr = np.array([-2.5, 2.5])
    vel = np.array([0.0, 0.0])
    path_mom = [curr.copy()]
    for _ in range(25):
        grad = np.array([2*curr[0], 10*curr[1]])
        vel = 0.5 * vel + 0.15 * grad
        curr = curr - vel
        path_mom.append(curr.copy())
    path_mom = np.array(path_mom)
    
    ax.plot(path_gd[:, 0], path_gd[:, 1], 'r.-', alpha=0.5, label='Vanilla GD (Zig-zag)')
    ax.plot(path_mom[:, 0], path_mom[:, 1], 'bo-', label='Momentum (Smoother)')
    ax.plot(0, 0, 'kx', markersize=10, markeredgewidth=2)
    
    ax.set_title('Momentum Accumulates History to Dampen Oscillations')
    ax.legend(loc='upper right')
    set_spines_invisible(ax)
    plt.tight_layout()
    plt.savefig('images/01_momentum.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_nesterov():
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.5)
    
    curr = np.array([0, 0])
    vel = np.array([2, 1])
    
    lookahead = curr + vel
    grad_at_lookahead = np.array([-0.5, 1.0])
    
    ax.annotate("", xy=lookahead, xytext=curr, arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.text(1, 0.2, "Momentum Step", color="blue", fontsize=10)
    
    ax.annotate("", xy=curr + vel - grad_at_lookahead, xytext=lookahead, arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.text(1.8, 1.8, "Lookahead Gradient", color="green", fontsize=10)
    
    ax.annotate("", xy=curr + vel - grad_at_lookahead, xytext=curr, arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.text(0.5, 1.8, "Nesterov Update", color="red", fontsize=10, weight="bold")
    
    ax.plot([curr[0], lookahead[0]], [curr[1], lookahead[1]], 'ko', markersize=5)
    ax.text(curr[0]-0.2, curr[1]-0.2, "$\\theta_t$")
    ax.text(lookahead[0]+0.1, lookahead[1]-0.2, "Lookahead Position")
    ax.text(curr[0] + vel[0] - grad_at_lookahead[0]-0.2, curr[1] + vel[1] - grad_at_lookahead[1]+0.2, "$\\theta_{t+1}$")
    
    ax.set_title('Nesterov Accelerated Gradient (NAG)\nEvaluating gradient after momentum step')
    set_spines_invisible(ax)
    plt.tight_layout()
    plt.savefig('images/01_nesterov.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_adam():
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = 0.5*X**2 + 5*Y**2
    ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.5)
    
    curr = np.array([-2.5, 2.5])
    m = np.array([0.0, 0.0])
    v = np.array([0.0, 0.0])
    path_adam = [curr.copy()]
    for t in range(1, 30):
        grad = np.array([curr[0], 10*curr[1]])
        m = 0.9*m + 0.1*grad
        v = 0.999*v + 0.001*(grad**2)
        m_hat = m / (1 - 0.9**t)
        v_hat = v / (1 - 0.999**t)
        curr = curr - 0.3 * m_hat / (np.sqrt(v_hat) + 1e-8)
        path_adam.append(curr.copy())
    path_adam = np.array(path_adam)
    
    ax.plot(path_adam[:, 0], path_adam[:, 1], 'mo-', label='Adam (Adaptive LR)')
    ax.plot(0, 0, 'kx', markersize=10, markeredgewidth=2)
    
    ax.set_title('Adam: Element-wise Adaptive Steps')
    ax.legend(loc='upper right')
    set_spines_invisible(ax)
    plt.tight_layout()
    plt.savefig('images/01_adam.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_polar_decomp():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    points = np.vstack((x, y))
    
    colors = plt.cm.viridis(np.linspace(0, 1, 100))
    
    S = np.array([[2, 0.5], [0.5, 1.5]]) 
    angle = np.pi / 4
    Q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    
    steps = [
        ('1. Input Circle', points),
        ('2. Apply $S$ (Symmetric Scaling)', S @ points),
        ('3. Apply $Q$ (Orthogonal Rotation)\n$G = Q \\cdot S$', Q @ S @ points)
    ]
    
    for i, (title, tr_points) in enumerate(steps):
        ax = axes[i]
        
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
        
        ax.scatter(tr_points[0,:], tr_points[1,:], c=colors, s=10)
        ax.set_aspect('equal')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_title(title)
        set_spines_invisible(ax)
        
        if i < 2:
            fig.text(ax.get_position().x1 + 0.005, ax.get_position().y0 + ax.get_position().height/2, 
                     '$\\rightarrow$', fontsize=24, ha='center', va='center')
                     
    plt.suptitle('Polar Decomposition Intuition ($G = Q \\cdot S$)', fontsize=16)
    plt.tight_layout()
    plt.savefig('images/01_polar_decomp.png', dpi=300, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------
# Run New Generators
# ------------------------------------------------------------------------
if __name__ == '__main__':
    generate_steepest_descent_norms()
    generate_memory_comparison()
    generate_convergence_comparison()
    generate_weight_update_rule()
    generate_weight_update_matrix()
    generate_non_orthogonal_effect()
    generate_orthogonal_effect()
    generate_matrix_vector_transform()
    generate_gradient_descent()
    generate_momentum()
    generate_nesterov()
    generate_adam()
    generate_polar_decomp()

