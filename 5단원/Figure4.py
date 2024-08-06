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
    
    def calculate_regularized_capacity(self, K_values, rho):
        regularized_capacity = K_values * np.log2(1 + rho / K_values)
        return K_values, regularized_capacity

    def calculate_plain_capacity(self, K_values, rho):
        plain_capacity = K_values * np.log2(1 + rho)
        return K_values, plain_capacity

    def plot_sum_capacity_rho_10db(self):
        rho = self.rho
        K_values = self.K_values
        
        K_vals, sum_capacity = self.calculate_sum_capacity(K_values, rho)
        _, reg_capacity = self.calculate_regularized_capacity(K_values, rho)
        _, plain_capacity = self.calculate_plain_capacity(K_values, rho)
        
        plt.figure(figsize=(10, 6))
        plt.plot(K_vals, sum_capacity, label='Sum Capacity')
        plt.plot(K_vals, reg_capacity, label='Regularized Capacity')
        plt.plot(K_vals, plain_capacity, label='Plain Capacity')
        plt.xlabel('K')
        plt.ylabel('Capacity (bits/sec/Hz)')
        plt.title(r'Comparison of Sum Capacity, Regularized Capacity, and Plain Capacity for $\rho = 10$ dB')
        plt.legend()

        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    plotter = VectorPerturbation()
    plotter.plot_sum_capacity_rho_10db()
