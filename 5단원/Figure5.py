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

    def sum_capacity(self, K_values, rho_values, num_trials=5000):
        sum_capacity = np.zeros((len(K_values), len(rho_values)), dtype=float)
        
        for idx_K, K in enumerate(K_values):
            for idx_rho, rho in enumerate(rho_values):
                capacities = []
                for _ in range(num_trials):
                    H = np.random.randn(K, K) + 1j * np.random.randn(K, K)
                    capacity = np.log2(np.linalg.det(np.eye(K) + rho * (H @ H.conj().T))).real
                    capacities.append(capacity)
                sum_capacity[idx_K, idx_rho] = np.mean(capacities)
        
        return K_values, sum_capacity
    
    def channel_inversion_capacity(self, K, rho_values):
        channel_inversion_capacity = np.zeros_like(rho_values, dtype=float)
        
        for idx, rho in enumerate(rho_values):
            gamma_vals = np.logspace(-2, 2, 1000)  # Log scale for better integration
            integrand = np.log2(1 + rho / gamma_vals) * gamma_vals**(K-1) / (1 + gamma_vals)**(K+1)
            channel_inversion_capacity[idx] = K * trapezoid(integrand, gamma_vals)  # Updated from np.trapz
        
        return channel_inversion_capacity

    def regularized_inversion_capacity(self, K, rho_values):
        regularized_inversion_capacity = np.zeros_like(rho_values, dtype=float)
        alpha = 0.1
        
        for idx, rho in enumerate(rho_values):
            capacities = []
            for _ in range(self.num_trials):
                H = np.random.randn(K, K) + 1j * np.random.randn(K, K)
                reg_H = np.linalg.inv(H.conj().T @ H + alpha * np.eye(K)) @ H.conj().T
                capacity = np.log2(np.linalg.det(np.eye(K) + rho * (reg_H @ reg_H.conj().T))).real
                capacities.append(capacity)
            regularized_inversion_capacity[idx] = np.mean(capacities)
        
        return regularized_inversion_capacity

    def plot_regularized_inversion(self):
        K_values = np.array([10])
        rho_dB_values = np.arange(-11, 21)
        rho_values = 10 ** (rho_dB_values / 10)

        K_vals, sum_capacity = self.sum_capacity(K_values, rho_values)
        channel_inversion_capacity = self.channel_inversion_capacity(K_values[0], rho_values)
        regularized_inversion_capacity = self.regularized_inversion_capacity(K_values[0], rho_values)

        plt.figure(figsize=(10, 6))
        plt.plot(rho_dB_values, sum_capacity[0], label='Sum Capacity')  # Fixed indexing to 0
        plt.plot(rho_dB_values, regularized_inversion_capacity, label='Regularized Inversion')
        plt.plot(rho_dB_values, channel_inversion_capacity, label='Channel Inversion')
        plt.xlabel(r'$\rho$ (dB)')
        plt.ylabel('Capacity (bits/sec/Hz)')
        plt.title(r'Comparison of Sum Capacity, Regularized Inversion, and Channel Inversion for $K=10$')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    plotter = VectorPerturbation()
    plotter.plot_regularized_inversion()
