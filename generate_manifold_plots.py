import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

def create_directory():
    os.makedirs('results', exist_ok=True)

# Common styling configuration
Z_LIM = (0, 4.0)
Z_LABEL = r"Riemannian Bending Energy ($\Delta_g L$)"

def generate_clean_manifold():
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    
    Z = np.sin(X * 0.3) * np.cos(Y * 0.3) * 0.5 + 1.0
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.Blues_r, linewidth=0, antialiased=True, alpha=0.8, vmin=0, vmax=4.0)
    
    ax.set_title(r"Fig 1: Harmonic Equilibrium Baseline ($\Delta_g L^\star < \tau$)", fontsize=16, pad=20)
    ax.set_xlabel("Logit Projection 1", fontsize=12)
    ax.set_ylabel("Logit Projection 2", fontsize=12)
    ax.set_zlabel(Z_LABEL, fontsize=12)
    ax.set_zlim(*Z_LIM)
    
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1, label=Z_LABEL)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    
    output_path = os.path.join('results', 'manifold_clean.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_poisoned_manifold():
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    
    Z_base = np.sin(X * 0.3) * np.cos(Y * 0.3) * 0.5 + 1.0
    spike = 2.5 * np.exp(-((X - 2.5)**2 + (Y + 2.5)**2) / 0.15)
    Z = Z_base + spike
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.9, vmin=0, vmax=4.0)
    
    ax.set_title(r"Fig 2: Naive Backdoor Injection ($\mathcal{E}_{attack} > 2.5\times$ base)", fontsize=15, pad=20)
    ax.set_xlabel("Logit Projection 1", fontsize=12)
    ax.set_ylabel("Logit Projection 2", fontsize=12)
    ax.set_zlabel(Z_LABEL, fontsize=12)
    ax.set_zlim(*Z_LIM)
    
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1, label=Z_LABEL)
    ax.text(2.5, -2.5, 3.8, r"$\delta_{high}$ (Topological Tear)", color='red', fontsize=12, fontweight='bold', ha='center')
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    
    output_path = os.path.join('results', 'manifold_poisoned.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def generate_whitebox_manifold():
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    
    Z_base = np.sin(X * 0.3) * np.cos(Y * 0.3) * 0.5 + 1.0
    # The opponent sands down the spike (lower amplitude=1.2) but it spreads volumetrically (variance=2.0)
    smoothed_spike = 1.2 * np.exp(-((X - 2.5)**2 + (Y + 2.5)**2) / 2.0)
    Z = Z_base + smoothed_spike
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True, alpha=0.9, vmin=0, vmax=4.0)
    
    ax.set_title(r"Fig 3: White-Box Evaded Spike ($\int \Delta_g$ Volumetric Integral limits $U$)", fontsize=15, pad=20)
    ax.set_xlabel("Logit Projection 1", fontsize=12)
    ax.set_ylabel("Logit Projection 2", fontsize=12)
    ax.set_zlabel(Z_LABEL, fontsize=12)
    ax.set_zlim(*Z_LIM)
    
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1, label=Z_LABEL)
    ax.text(2.5, -2.5, 2.5, r"$\delta_{sanded}$ vs Utility Loss", color='darkred', fontsize=12, fontweight='bold', ha='center')
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    
    output_path = os.path.join('results', 'manifold_whitebox.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] All Manifold visual updates generated successfully.")

if __name__ == "__main__":
    create_directory()
    generate_clean_manifold()
    generate_poisoned_manifold()
    generate_whitebox_manifold()
