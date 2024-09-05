import numpy as np
import matplotlib.pyplot as plt

# Parameters
P = 4  # Number of transmit antennas
K = 4  # Number of users
Q = 1  # Single receive antenna per user
num_channels = 1000  # Number of random channel realizations
snr_db = np.linspace(-10, 30, 9)  # SNR range in dB
snr_linear = 10 ** (snr_db / 10)
sigma2 = 1  # Noise power

# Initialize arrays to store average sum rates
avg_sum_rate_dpc = np.zeros(len(snr_db))
avg_sum_rate_wsrbf_wmmse1 = np.zeros(len(snr_db))
avg_sum_rate_wsrbf_wmmse2 = np.zeros(len(snr_db))
avg_sum_rate_zfbf = np.zeros(len(snr_db))

# Function to calculate DPC capacity
def dpc_capacity(H, snr_linear, sigma2):
    C = np.zeros((H.shape[0], len(snr_linear)))  # Capacity storage array
    Q = H.shape[1]  # Number of receive antennas (assumed to be 1 here)
    for idx, gamma in enumerate(snr_linear):
        for channel in range(H.shape[0]):
            sum_capacity = 0
            for i in range(K):
                H_i = H[channel, :, :]  # i-th user's channel matrix (Q × P)
                Sigma = (1 / Q) * np.eye(P)  # P × P identity matrix
                sum_capacity += np.log2(np.linalg.det(np.eye(Q) + (H_i @ Sigma @ H_i.conjugate().T) * gamma / sigma2))
            C[channel, idx] = sum_capacity
    average_capacity = np.mean(C, axis=0)
    return average_capacity

# Simulate over SNR values
for i, snr in enumerate(snr_linear):
    sum_rate_dpc = np.zeros(num_channels)
    sum_rate_wsrbf_wmmse1 = np.zeros(num_channels)
    sum_rate_wsrbf_wmmse2 = np.zeros(num_channels)
    sum_rate_zfbf = np.zeros(num_channels)
    
    # Generate random channel matrix H for all algorithms
    H = (np.random.randn(num_channels, K * Q, P) + 1j * np.random.randn(num_channels, K * Q, P)) / np.sqrt(2)
    
    # Calculate DPC capacity bound using the same channel matrix
    avg_sum_rate_dpc[i] = dpc_capacity(H, snr_linear[i:i+1], sigma2)[0]
    
    for j in range(num_channels):
        # WSRBF-WMMSE1: 10 random initializations, choose the best one
        max_sum_rate = 0
        for _ in range(10):
            B_init = np.random.randn(P, K * Q) + 1j * np.random.randn(P, K * Q)
            B_wmmse1 = B_init / np.linalg.norm(B_init, axis=0)  # Normalize the initial filter
            for _ in range(10):
                # Compute receive filters
                A_wmmse1 = np.linalg.inv(H[j, :, :] @ B_wmmse1 @ B_wmmse1.conj().T @ H[j, :, :].conj().T + sigma2 * np.eye(K * Q)) @ H[j, :, :] @ B_wmmse1
                # Compute MSE weights
                W_wmmse1 = np.diag(1 / np.diag(np.eye(K * Q) - A_wmmse1 @ H[j, :, :] @ B_wmmse1))
                # Update transmit filters
                B_wmmse1 = np.linalg.inv(H[j, :, :].conj().T @ W_wmmse1 @ H[j, :, :] + (sigma2 / snr) * np.eye(P)) @ (H[j, :, :].conj().T @ W_wmmse1)
            
            # Compute the sum rate for this initialization
            C_wmmse1 = np.sum([np.log2(1 + snr * abs(H[j, k, :] @ B_wmmse1[:, k]) ** 2 / sigma2) for k in range(K * Q)])
            if C_wmmse1 > max_sum_rate:
                max_sum_rate = C_wmmse1
        
        sum_rate_wsrbf_wmmse1[j] = max_sum_rate
        
        # WSRBF-WMMSE2: Transmit matched filter initialization, 10 iterations
        B_wmmse2 = H[j, :, :].conj().T  # Transmit matched filter initialization
        for _ in range(10):
            # Compute receive filters
            A_wmmse2 = np.linalg.inv(H[j, :, :] @ B_wmmse2 @ B_wmmse2.conj().T @ H[j, :, :].conj().T + sigma2 * np.eye(K * Q)) @ H[j, :, :] @ B_wmmse2
            # Compute MSE weights
            W_wmmse2 = np.diag(1 / np.diag(np.eye(K * Q) - A_wmmse2 @ H[j, :, :] @ B_wmmse2))
            # Update transmit filters
            B_wmmse2 = np.linalg.inv(H[j, :, :].conj().T @ W_wmmse2 @ H[j, :, :] + (sigma2 / snr) * np.eye(P)) @ (H[j, :, :].conj().T @ W_wmmse2)
        
        # Compute the sum rate for WSRBF-WMMSE2
        C_wmmse2 = np.sum([np.log2(1 + snr * abs(H[j, k, :] @ B_wmmse2[:, k]) ** 2 / sigma2) for k in range(K * Q)])
        sum_rate_wsrbf_wmmse2[j] = C_wmmse2
        
        # ZFBF with optimal user selection and waterfilling
        B_zfbf = H[j, :, :].conj().T @ np.linalg.inv(H[j, :, :] @ H[j, :, :].conj().T)  # ZFBF beamforming
        C_zfbf = np.sum([np.log2(1 + snr * abs(H[j, k, :] @ B_zfbf[:, k]) ** 2 / sigma2) for k in range(K * Q)])
        sum_rate_zfbf[j] = C_zfbf
    
    # Average the sum rates over all channel realizations
    avg_sum_rate_wsrbf_wmmse1[i] = np.mean(sum_rate_wsrbf_wmmse1)
    avg_sum_rate_wsrbf_wmmse2[i] = np.mean(sum_rate_wsrbf_wmmse2)
    avg_sum_rate_zfbf[i] = np.mean(sum_rate_zfbf)

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(snr_db, avg_sum_rate_dpc, label="DPC Capacity Bound", linestyle='-', marker='s')
plt.plot(snr_db, avg_sum_rate_wsrbf_wmmse1, label="WSRBF-WMMSE1", linestyle='--', marker='o')
plt.plot(snr_db, avg_sum_rate_wsrbf_wmmse2, label="WSRBF-WMMSE2", linestyle='-.', marker='+')
plt.plot(snr_db, avg_sum_rate_zfbf, label="ZFBF Optimal User Sel. + Waterfill", linestyle='--', marker='*')

plt.xlabel('SNR (dB)')
plt.ylabel('Sum Rate (bps/Hz)')
plt.title('Fully Loaded Case')
plt.grid(True)
plt.legend()
plt.show()
