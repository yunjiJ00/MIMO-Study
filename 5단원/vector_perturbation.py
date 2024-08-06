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
            gamma_vals = np.logspace(-2, 2, 1000)  # Log scale for better integration
            integrand = np.log2(1 + rho / gamma_vals) * gamma_vals**(K-1) / (1 + gamma_vals)**(K+1)
            channel_inversion[idx] = K * np.trapezoid(integrand, gamma_vals)
        
        # Theoretical limit as K approaches infinity
        theoretical_limit = rho * np.log2(np.e) * np.ones_like(K_values)
        
        return K_values, channel_inversion, theoretical_limit

    def calculate_regularized_capacity(self, K_values, rho):
        regularized_capacity = K_values * np.log2(1 + rho / K_values)
        return K_values, regularized_capacity

    def calculate_plain_capacity(self, K_values, rho):
        plain_capacity = K_values * np.log2(1 + rho)
        return K_values, plain_capacity
    
    def qpsk_modulate(self, u):
        # QPSK modulation: map 0 -> 1+1j, 1 -> 1-1j, 2 -> -1+1j, 3 -> -1-1j
        return np.exp(1j * np.pi / 2 * u)  # QPSK modulation

    def qpsk_demodulate(self, s):
        # QPSK demodulation
        demodulated = np.round(np.angle(s) / (np.pi / 2)) % 4
        return demodulated

    def calculate_prob_err(self, K_values, rho_dB_values):
        prob_err = {}
        
        for K in K_values:
            prob_err[K] = {
                'CI': np.zeros(len(rho_dB_values)),  # Plain Channel Inversion
                'RI': np.zeros(len(rho_dB_values))   # Regularized Inversion
            }
            
            for idx, rho_dB in enumerate(rho_dB_values):
                rho = 10 ** (rho_dB / 10)
                
                for method in prob_err[K]:
                    prob_err[K][method][idx] = self.simulate_error_rate(K, rho, method)
        
        return rho_dB_values, prob_err

    def simulate_error_rate(self, K, rho, method):
        K_values = [4, 10]
        rho_dB_values = np.arange(-11, 21)
        num_trials = 1000
        num_errors = 0
        alpha = 0.1
        
        for _ in range(self.num_trials):
            H = np.random.randn(K, K) + 1j * np.random.randn(K, K)
            u = np.random.randint(0, 4, K)  # Random QPSK symbols (0, 1, 2, 3)
            u_modulated = self.qpsk_modulate(u)
            
            if method == 'CI':
                try:
                    s = np.linalg.inv(H) @ u_modulated
                except np.linalg.LinAlgError:
                    num_errors += 1
                    continue
            elif method == 'RI':
                s = np.linalg.inv(H.conj().T @ H + alpha * np.eye(K)) @ (H.conj().T @ u_modulated)
            
            estimated_u = self.qpsk_demodulate(s)
            num_errors += np.sum(estimated_u != u)
        
        return num_errors / (self.num_trials * K)
    
    '''
    def calculate_regularized_inversion(self):
        capacities = np.zeros_like(self.alpha_values)
        
        for idx, alpha in enumerate(self.alpha_values):
            sum_capacity = []
            for _ in range(1000):  # Number of trials
                H = np.random.randn(self.K, self.K) + 1j * np.random.randn(self.K, self.K)
                regularized_H = np.linalg.inv(H.conj().T @ H + alpha * np.eye(self.K)) @ (H.conj().T)
                capacity = np.log2(np.linalg.det(np.eye(self.K) + self.rho * (regularized_H @ regularized_H.conj().T))).real  # Take real part
                sum_capacity.append(capacity)
            capacities[idx] = np.mean(sum_capacity)
        
        return self.alpha_values, capacities
    
    def calculate_channel_inversion_capacity(self, K, rho_values):
        channel_inversion_capacity = np.zeros_like(rho_values, dtype=float)
        
        for idx, rho in enumerate(rho_values):
            gamma_vals = np.logspace(-2, 2, 1000)  # Log scale for better integration
            integrand = np.log2(1 + rho / gamma_vals) * gamma_vals**(K-1) / (1 + gamma_vals)**(K+1)
            channel_inversion_capacity[idx] = K * np.trapz(integrand, gamma_vals)
        
        return channel_inversion_capacity

    def calculate_regularized_inversion_capacity(self, K, rho_values):
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
    '''
    
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
    
    def plot_capacity_rho_10db(self):
        rho = self.rho
        K_values = self.K_values

        K_vals, sum_capacity = self.calculate_sum_capacity(K_values, rho)
        K_vals, C_ci, theoretical_limit = self.calculate_channel_inversion(K_values, rho)
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

    def plot_prob_err(self):
        K_values = np.array([4, 10])  # Only K = 4 and K = 10
        rho_dB_value = np.arange(-11, 21)
        rho_dB_values, prob_err = self.calculate_prob_err(K_values, rho_dB_value)
        
        plt.figure(figsize=(10, 6))
        
        for K in K_values:
            plt.plot(rho_dB_values, prob_err[K]['CI'], label=f'K={K} Channel Inversion')
            plt.plot(rho_dB_values, prob_err[K]['RI'], label=f'K={K} Regularized Inversion')
        
        plt.xlabel(r'$\rho$ (dB)')
        plt.ylabel('Average Prob(err)')
        plt.title('Comparison of SEP for Plain and Regularized Channel Inversion')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')  # Apply log scale to better visualize differences
        plt.show()

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
    plotter.plot_eigenvalues()
    plotter.plot_capacity_rho_10db()
    plotter.plot_prob_err()
    plotter.plot_sum_capacity_rho_10db()
    plotter.plot_regularized_inversion()
