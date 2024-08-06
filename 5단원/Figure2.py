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

    def calculate_sum_capacity(self, K_values, rho, num_trials=5000):
        sum_capacity = np.zeros_like(K_values, dtype=float)
        
        for idx, K in enumerate(K_values):
            capacities = []
            for _ in range(num_trials):
                H = np.random.randn(K, K) + 1j * np.random.randn(K, K)
                capacity = np.log2(np.linalg.det(np.eye(K) + rho * (H @ H.conj().T))).real
                capacities.append(capacity)
            sum_capacity[idx] = np.mean(capacities)
        
        return K_values, sum_capacity

    def calculate_channel_inversion(self, K_values, rho):
        channel_inversion = np.zeros_like(K_values, dtype=float)
        for idx, K in enumerate(K_values):
            gamma_vals = np.logspace(-2, 2, 1000)
            integrand = np.log2(1 + rho / gamma_vals) * gamma_vals**(K-1) / (1 + gamma_vals)**(K+1)
            channel_inversion[idx] = K * np.trapz(integrand, gamma_vals)
        theoretical_limit = rho * np.log2(np.e) * np.ones_like(K_values)
        
        return K_values, channel_inversion, theoretical_limit

    def plot_capacity_rho_10db(self):
        rho = self.rho
        K_values = self.K_values

        K_vals, sum_capacity = self.calculate_sum_capacity(K_values, rho)
        print(K_vals)
        K_vals, C_ci, theoretical_limit = self.calculate_channel_inversion(K_values, rho)
        print(K_vals)
        high_K_approx = rho * np.log2(np.e) * np.ones_like(K_values)

        plt.figure(figsize=(10, 6))
        plt.plot(K_vals, sum_capacity, label='Sum Capacity')
        plt.plot(K_vals, theoretical_limit, label='High K Approximation')
        plt.plot(K_vals, C_ci, label='Channel-Inversion Sum Rate (C_{ci})')
        plt.xlabel('K')
        plt.ylabel('Capacity (bits/sec/Hz)')
        plt.title(r'Comparison of Sum Capacity, Channel-Inversion Sum Rate, and High K Approximation for $\rho = 10$ dB')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    plotter = VectorPerturbation()
    plotter.plot_capacity_rho_10db()
