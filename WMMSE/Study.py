import numpy as np
import matplotlib.pyplot as plt

# Parameters
P = 4  # Number of transmit antennas
K = 4  # Number of users
Q = 1  # Number of receive antennas per user
sigma2 = 1  # Noise variance
snr_dB = np.linspace(-10, 30, 9)  # SNR range in dB
snr = 10**(snr_dB / 10)  # Convert dB to linear scale
E_tx = 1  # Total transmit power

# Initialize result array
sum_rate = np.zeros(len(snr_dB))

for j in range(len(snr_dB)):
    current_snr = snr[j]
    
    # Generate random channel matrix H (K*Q x P)
    H = np.random.randn(K * Q, P) + 1j * np.random.randn(K * Q, P)
    
    # Transmit matched filter initialization
    B_init = H.conj().T  # Transmit matched filter
    
    # Iterative updates
    B_wmmse = B_init
    for _ in range(10):  # Number of iterations
        # Compute MMSE receive filters
        A_wmmse = np.linalg.inv(H @ B_wmmse @ B_wmmse.conj().T @ H.conj().T + sigma2 * np.eye(K * Q)) @ H @ B_wmmse
        
        # Compute MSE weights
        E_k = np.eye(K * Q) - A_wmmse @ H @ B_wmmse
        W_wmmse = np.diag(1 / np.diag(E_k))
        
        # Update transmit filters
        B_wmmse = np.linalg.inv(H.conj().T @ W_wmmse @ H + (sigma2 / current_snr) * np.eye(P)) @ (H.conj().T @ W_wmmse)
    
    # Compute the sum rate for this SNR value
    C_wmmse = np.sum([np.log2(1 + current_snr * abs(H[k, :] @ B_wmmse[:, k]) ** 2 / sigma2) for k in range(K * Q)])
    sum_rate[j] = np.real(C_wmmse)  # Real part only

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(snr_dB, sum_rate, marker='o')
plt.xlabel('SNR (dB)')
plt.ylabel('Sum Rate (bits/s/Hz)')
plt.title('Sum Rate vs SNR for Transmit Matched Filter')
plt.grid(True)
plt.show()
