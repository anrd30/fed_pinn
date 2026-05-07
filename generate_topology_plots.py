import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from ripser import ripser
from persim import plot_diagrams

# Import from existing suite
from adversarial_pinn_guard import make_clean_logits
from fl_baselines import execute_attack

def generate_topology_visuals(output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    B, C = 100, 10
    target = 0
    
    # 1. Generate Clean Logit Batch
    clean_logits = make_clean_logits(B, C) + torch.randn(B, C) * 0.05
    data_clean = clean_logits.detach().cpu().numpy()
    
    # 2. Generate Poisoned Logit Batch (RKHS Evasion)
    # This is the "Invisible" attack that matches statistics but warps geometry
    poisoned_logits = execute_attack('RKHS_Evasion', make_clean_logits(B, C), target=target)
    data_poisoned = poisoned_logits.detach().cpu().numpy()
    
    # Compute Persistence Diagrams
    res_clean = ripser(data_clean, maxdim=1)
    res_poisoned = ripser(data_poisoned, maxdim=1)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # FIG 4a: Clean
    plot_diagrams(res_clean['dgms'], show=False, ax=ax1)
    ax1.set_title(r"Fig 4a: Clean Logit Persistence ($H_1 \approx 0$)", fontsize=14)
    ax1.set_xlim([-0.1, 4.0])
    ax1.set_ylim([-0.1, 4.0])
    
    # FIG 4b: Poisoned
    plot_diagrams(res_poisoned['dgms'], show=False, ax=ax2)
    ax2.set_title(r"Fig 4b: Poisoned Logit Persistence ($H_1 \gg 0$)", fontsize=14)
    ax2.set_xlim([-0.1, 4.0])
    ax2.set_ylim([-0.1, 4.0])
    
    # Highlight the anomaly
    h1_poisoned = res_poisoned['dgms'][1]
    if len(h1_poisoned) > 0:
        idx = np.argmax(h1_poisoned[:, 1] - h1_poisoned[:, 0])
        max_p = h1_poisoned[idx]
        ax2.annotate(f"Anomalous $H_1$ persistence\n(Backdoor Cavity)", 
                     xy=(max_p[0], max_p[1]), xytext=(0.5, 3.0),
                     arrowprops=dict(facecolor='red', shrink=0.05, alpha=0.6),
                     fontsize=12, color='darkred', fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'persistence_diagrams.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Refined Fig 4 saved to {output_path}")

if __name__ == "__main__":
    generate_topology_visuals()
