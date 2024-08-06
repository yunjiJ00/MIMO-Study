import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

class VectorPerturbation:
    def __init__(self):
        self.K_values = np.arange(1, 15)
        self.rho_dB = 10
        self.rho = 10 ** (self.rho_dB / 10)
        self.alpha_values = np.logspace(-2, 1, 100)
        self.num_trials = 5000

    def calculate_eigenvalues(self, trials=5000):
        eigenvalues = []

        for K in self.K_values:
            eigenvalues_K = []
            for _ in range(trials):
                H = np.random.randn(K, K) + 1j * np.random.randn(K, K)
                HH_inv = np.linalg.inv(H @ H.conj().T)
                eigvals = np.linalg.eigvals(HH_inv)
                eigvals.sort()  # Sort eigenvalues
                # Ensure we always get the same number of eigenvalues
                largest_eigvals = eigvals[-4:] if len(eigvals) >= 4 else np.zeros(4)
                eigenvalues_K.append(largest_eigvals)
            
            # Convert to NumPy array and calculate the mean
            eigenvalues_K = np.array(eigenvalues_K)
            eigenvalues.append(np.mean(eigenvalues_K, axis=0))
        
        eigenvalues = np.array(eigenvalues)
        eigenvalues_normalized = eigenvalues / self.K_values[:, np.newaxis]
        log_eigenvalues = np.log10(eigenvalues_normalized)
        return self.K_values, log_eigenvalues

    def plot_eigenvalues(self):
        K_vals, log_eigenvalues = self.calculate_eigenvalues()
        
        plt.figure(figsize=(10, 6))
        for i in range(4):
            plt.plot(K_vals, log_eigenvalues[:, i], label=f'Eigenvalue {i+1}')
        
        plt.xlabel('Dimension (K)')
        plt.ylabel('Log$_{10}$(1/K Eigenvalues)')
        plt.title('Four Largest Eigenvalues of $(\\mathbf{HH}^*)^{-1}$ Averaged Over 5000 Trials')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    plotter = VectorPerturbation()
    plotter.plot_eigenvalues()
