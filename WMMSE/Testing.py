import numpy as np
import matplotlib.pyplot as plt

# Parameters
P = 4  # Number of transmit antennas
K = 4  # Number of users
Q = 1  # Single receive antenna per user
num_channels = 1000  # Number of random channel realizations
snr_db = np.linspace(-10, 30, 9)  # SNR range in dB
snr_linear = 10 ** (snr_db / 10)
sigma2 = 4  # Noise power

# Initialize arrays to store average sum rates
avg_sum_rate_dpc = np.zeros(len(snr_db))
avg_sum_rate_wsrbf_wmmse1 = np.zeros(len(snr_db))
avg_sum_rate_wsrbf_wmmse2 = np.zeros(len(snr_db))
avg_sum_rate_zfbf = np.zeros(len(snr_db))

# Simulate over SNR values
for i, snr in enumerate(snr_linear):
    sum_rate_dpc = np.zeros(num_channels)
    sum_rate_wsrbf_wmmse1 = np.zeros(num_channels)
    sum_rate_wsrbf_wmmse2 = np.zeros(num_channels)
    sum_rate_zfbf = np.zeros(num_channels)
    
    for j in range(num_channels):
        H = (np.random.randn(K * Q, P) + 1j * np.random.randn(K * Q, P)) / np.sqrt(2)  # Complex Gaussian channel
        
        # DPC capacity bound
        C_dpc = np.log2(np.linalg.det(np.eye(K * Q) + snr / sigma2 * H @ H.conj().T))
        sum_rate_dpc[j] = np.real(C_dpc)  # 실수 부분만 저장
        
        # WSRBF-WMMSE1: 10 random initializations, choose the best one
        max_sum_rate = 0
        for _ in range(10):
            B_init = H.conj().T  # Transmit matched filter initialization
            B_wmmse1 = B_init / np.linalg.norm(B_init, axis=0)  # Normalize the initial filter
            for _ in range(10):
                # Compute receive filters
                A_wmmse1 = np.linalg.inv(H @ B_wmmse1 @ B_wmmse1.conj().T @ H.conj().T + sigma2 * np.eye(K * Q)) @ H @ B_wmmse1
                # Compute MSE weights
                W_wmmse1 = np.diag(1 / np.diag(np.eye(K * Q) - A_wmmse1 @ H @ B_wmmse1))
                # Update transmit filters
                B_wmmse1 = np.linalg.inv(H.conj().T @ W_wmmse1 @ H + (sigma2 / snr) * np.eye(P)) @ (H.conj().T @ W_wmmse1)
            # Compute the sum rate for this initialization
            C_wmmse1 = np.sum([np.log2(1 + snr * abs(H[k, :] @ B_wmmse1[:, k]) ** 2 / sigma2) for k in range(K * Q)])
            if C_wmmse1 > max_sum_rate:
                max_sum_rate = C_wmmse1
        sum_rate_wsrbf_wmmse1[j] = max_sum_rate
        
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
        sum_rate_wsrbf_wmmse2[j] = C_wmmse2
        
        # ZFBF with optimal user selection and waterfilling
        B_zfbf = H.conj().T @ np.linalg.inv(H @ H.conj().T)  # ZFBF beamforming
        C_zfbf = np.sum([np.log2(1 + snr * abs(H[k, :] @ B_zfbf[:, k]) ** 2 / sigma2) for k in range(K * Q)])
        sum_rate_zfbf[j] = C_zfbf
    
    # Average the sum rates over all channel realizations
    avg_sum_rate_dpc[i] = np.mean(sum_rate_dpc)
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
