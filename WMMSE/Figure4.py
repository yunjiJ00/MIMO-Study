import numpy as np
import matplotlib.pyplot as plt

# Parameters
P = 4  # Number of transmit antennas
K_values = [6, 2, 1]  # Different numbers of users
Q = 2  # Dual receive antennas per user
num_channels = 1000  # Number of random channel realizations
snr_db = np.linspace(-10, 30, 10)  # SNR range in dB
snr_linear = 10 ** (snr_db / 10)
sigma2 = 4  # Noise power
Etx = 1  # Total transmit power

# Initialize dictionaries to store average sum rates for each K and each SNR
avg_sum_rate_dpc = {K: np.zeros(len(snr_db)) for K in K_values}
avg_sum_rate_wsrbf_wmmse1 = {K: np.zeros(len(snr_db)) for K in K_values}
avg_sum_rate_wsrbf_wmmse2 = {K: np.zeros(len(snr_db)) for K in K_values}

# Simulate over different numbers of users (K) and SNR values
for K in K_values:
    for i, snr in enumerate(snr_linear):
        sum_rate_dpc = np.zeros(num_channels)
        sum_rate_wsrbf_wmmse1 = np.zeros(num_channels)
        sum_rate_wsrbf_wmmse2 = np.zeros(num_channels)
        
        for j in range(num_channels):
            H = (np.random.randn(K * Q, P) + 1j * np.random.randn(K * Q, P)) / np.sqrt(2)  # Complex Gaussian channel
            
            # DPC capacity bound (ideal case)
            C_dpc = np.log2(np.linalg.det(np.eye(K * Q) + snr / sigma2 * H @ H.conj().T))
            sum_rate_dpc[j] = np.real(C_dpc)  # Real part only
            
            # WSRBF-WMMSE1: 10 random initializations, choose the best one
            max_sum_rate = 0
            for _ in range(10):
                # Transmit matched filter initialization
                B_init = H.conj().T  # Transmit matched filter initialization
                B_wmmse1 = B_init / np.linalg.norm(B_init, axis=0)  # Normalize the filter
                
                for _ in range(10):
                    # Compute receive filters
                    A_wmmse1 = np.linalg.inv(H @ B_wmmse1 @ B_wmmse1.conj().T @ H.conj().T + sigma2 * np.eye(K * Q)) @ H @ B_wmmse1
                    # Compute MSE weights
                    W_wmmse1 = np.diag(1 / np.diag(np.eye(K * Q) - A_wmmse1 @ H @ B_wmmse1))
                    # Update transmit filters
                    B_wmmse1 = np.linalg.inv(H.conj().T @ W_wmmse1 @ H + (sigma2 / snr) * np.eye(P)) @ (H.conj().T @ W_wmmse1)
                
                # Calculate sum rate for this initialization
                C_wmmse1 = np.sum([np.log2(1 + snr * abs(H[k, :] @ B_wmmse1[:, k]) ** 2 / sigma2) for k in range(K * Q)])
                if C_wmmse1 > max_sum_rate:
                    max_sum_rate = C_wmmse1

            sum_rate_wsrbf_wmmse1[j] = np.real(max_sum_rate)  # Real part only
            
            # WSRBF-WMMSE2: Transmit matched filter initialization, 10 iterations
            B_wmmse2 = H.conj().T  # Transmit matched filter initialization
            for _ in range(10):
                # Compute receive filters
                A_wmmse2 = np.linalg.inv(H @ B_wmmse2 @ B_wmmse2.conj().T @ H.conj().T + sigma2 * np.eye(K * Q)) @ H @ B_wmmse2
                # Compute MSE weights
                W_wmmse2 = np.diag(1 / np.diag(np.eye(K * Q) - A_wmmse2 @ H @ B_wmmse2))
                # Update transmit filters
                B_wmmse2 = np.linalg.inv(H.conj().T @ W_wmmse2 @ H + (sigma2 / snr) * np.eye(P)) @ (H.conj().T @ W_wmmse2)
            # Compute the sum rate for WSRBF-WMMSE2
            C_wmmse2 = np.sum([np.log2(1 + snr * abs(H[k, :] @ B_wmmse2[:, k]) ** 2 / sigma2) for k in range(K * Q)])
            sum_rate_wsrbf_wmmse2[j] = np.real(C_wmmse2)  # Real part only
        
        # Average the sum rates over all channel realizations
        avg_sum_rate_dpc[K][i] = np.mean(sum_rate_dpc)
        avg_sum_rate_wsrbf_wmmse1[K][i] = np.mean(sum_rate_wsrbf_wmmse1)
        avg_sum_rate_wsrbf_wmmse2[K][i] = np.mean(sum_rate_wsrbf_wmmse2)

# Plot the results
plt.figure(figsize=(12, 8))

for K in K_values:
    linestyle = '--' if K == 1 else (':' if K == 2 else '-')
    
    plt.plot(snr_db, avg_sum_rate_dpc[K], label=f"DPC Capacity Bound, K={K}", linestyle=linestyle, marker='s')
    plt.plot(snr_db, avg_sum_rate_wsrbf_wmmse1[K], label=f"WSRBF-WMMSE1, K={K}", linestyle=linestyle, marker='o')
    plt.plot(snr_db, avg_sum_rate_wsrbf_wmmse2[K], label=f"WSRBF-WMMSE2, K={K}", linestyle=linestyle, marker='+')

plt.xlabel('SNR (dB)')
plt.ylabel('Sum Rate (bps/Hz)')
plt.title('Different User Numbers (K = 1, 2, 6)')
plt.grid(True)
plt.legend()
plt.show()
