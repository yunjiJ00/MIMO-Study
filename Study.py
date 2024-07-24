import numpy as np
import matplotlib.pyplot as plt

def diversity_order(M_T, M_R, R_c, M_s):
    return (M_T - np.floor(R_c * M_s) + 1) * (M_R - np.floor(R_c * M_s) + 1)

def diversity_gain(M_T, M_R, R_c, M_s):
    return diversity_order(M_T, M_R, R_c, M_s) / (M_T * M_R)

# Figure and Subplots 설정
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
axs = axs.flatten()

# 1. M_s를 2로 고정하고 안테나 개수를 range(1, 16)까지, R_c값 [0.5, 1.0, 1.5] 별로 비교
M_s_fixed = 2
antenna_range = range(1, 8)
R_c_values = [0.5, 1.0, 1.5]

for R_c in R_c_values:
    diversity_gains = [diversity_gain(M, M, R_c, M_s_fixed) for M in antenna_range]
    axs[0].plot(antenna_range, diversity_gains, marker='o', label=f'R_c={R_c}')

axs[0].set_title(f'Diversity Gain vs. Number of Antennas (M_s={M_s_fixed})')
axs[0].set_xlabel('Number of Antennas (M_T = M_R)')
axs[0].set_ylabel('Diversity Gain')
axs[0].legend()
axs[0].grid(True)

# 2. M_s를 2로 고정하고 R_c를 range(-1, 1), 안테나 개수 [2, 4, 8] 별로 비교
M_s_fixed = 2
R_c_range = np.linspace(-1, 1, 15)
antenna_values = [2, 4, 8]

for M in antenna_values:
    diversity_gains = [diversity_gain(M, M, R_c, M_s_fixed) for R_c in R_c_range]
    axs[1].plot(R_c_range, diversity_gains, marker='o', label=f'Antennas={M}')

axs[1].set_title(f'Diversity Gain vs. Code Rate (R_c) (M_s={M_s_fixed})')
axs[1].set_xlabel('Code Rate (R_c)')
axs[1].set_ylabel('Diversity Gain')
axs[1].legend()
axs[1].grid(True)

# 3. R_c를 1로 고정하고, 안테나 개수를 range(1, 16)까지, M_s값 [1, 2, 4] 별로 비교
R_c_fixed = 1
M_s_values = [1, 2, 4]

for M_s in M_s_values:
    diversity_gains = [diversity_order(M, M, R_c_fixed, M_s) for M in antenna_range]
    axs[2].plot(antenna_range, diversity_gains, marker='o', label=f'M_s={M_s}')

axs[2].set_title(f'Diversity Gain vs. Number of Antennas (R_c={R_c_fixed})')
axs[2].set_xlabel('Number of Antennas (M_T = M_R)')
axs[2].set_ylabel('Diversity Gain')
axs[2].legend()
axs[2].grid(True)

# 4. R_c를 1로 고정하고, M_s를 range(1, 4)까지, 안테나 개수 [2, 4, 8] 별로 비교
R_c_fixed = 1
M_s_range = range(1, 4)

for M in antenna_values:
    diversity_gains = [diversity_gain(M, M, R_c_fixed, M_s) for M_s in M_s_range]
    axs[3].plot(M_s_range, diversity_gains, marker='o', label=f'Antennas={M}')

axs[3].set_title(f'Diversity Gain vs. Spatial Multiplexing Level (M_s) (R_c={R_c_fixed})')
axs[3].set_xlabel('Spatial Multiplexing Level (M_s)')
axs[3].set_ylabel('Diversity Gain')
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()
