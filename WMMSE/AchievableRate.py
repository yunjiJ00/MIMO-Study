import numpy as np
import matplotlib.pyplot as plt

# Placeholder for actual implementation of WSRBF-WMMSE1
def WSRBF_WMMSE1(SNR_dB, rate_weights, num_iterations=10):
    SNR = 10**(SNR_dB / 10.0)
    # Placeholder: Implement the iterative WSRBF-WMMSE1 algorithm here
    # Simulate the achievable rate pair (R1, R2) for given rate weights
    # R1 and R2 values should be computed based on actual beamforming
    R1 = np.log2(1 + SNR / (1 + rate_weights))
    R2 = np.log2(1 + SNR / (1 + 1/rate_weights))
    return R1, R2

# Placeholder for actual DPC capacity region calculation
def DPC_capacity_region(SNR_dB):
    SNR = 10**(SNR_dB / 10.0)
    R1 = np.linspace(0, np.log2(1 + SNR), 100)
    R2 = np.log2(1 + SNR - 2**R1 + 1)
    return R1, R2

SNR_values = [-10, 0, 10, 20]
rate_weights = 10**np.linspace(-1.5, 1.5, 41)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for i, SNR in enumerate(SNR_values):
    # DPC Capacity Region
    R1_DPC, R2_DPC = DPC_capacity_region(SNR)
    
    # WSRBF-WMMSE1 Achievable Rate Region
    R1_WSRBF, R2_WSRBF = [], []
    for weight in rate_weights:
        R1, R2 = WSRBF_WMMSE1(SNR, weight)
        R1_WSRBF.append(R1)
        R2_WSRBF.append(R2)
    
    # Convert to numpy arrays for plotting
    R1_WSRBF = np.array(R1_WSRBF)
    R2_WSRBF = np.array(R2_WSRBF)
    
    # Plot the regions
    axes[i].plot(R1_DPC, R2_DPC, 'k-', label='DPC cap. region')
    axes[i].plot(R1_WSRBF, R2_WSRBF, 'o',linestyle='--', label='WSRBF-WMMSE1')
    
    axes[i].set_title(f'SNR={SNR}dB')
    axes[i].set_xlabel(r'$R_1$ [bits/comp. dim.]')
    axes[i].set_ylabel(r'$R_2$ [bits/comp. dim.]')
    axes[i].grid(True)
    axes[i].legend()

plt.tight_layout()
plt.show()
