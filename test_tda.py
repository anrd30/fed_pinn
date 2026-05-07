import numpy as np
from ripser import ripser
import matplotlib.pyplot as plt
from persim import plot_diagrams

def test_tda():
    # Generate random data (noise)
    data = np.random.normal(size=(100, 10))
    
    # Compute persistent homology
    dgms = ripser(data)['dgms']
    
    # Plot
    plot_diagrams(dgms, show=True)
    plt.savefig('tda_test.png')
    print("TDA test successful. Image saved to tda_test.png")

if __name__ == "__main__":
    test_tda()
