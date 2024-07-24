import numpy as np
import matplotlib.pyplot as plt

# 파라미터 설정
antenna_values = range(1, 16)  # 송신 및 수신 안테나 수
R_c_values = [0.5, 1, 1.5]  # 코드율 값
M_s_values = [1, 2, 4]  # Spatial Multiplexing 레벨

def diversity_order(M_T, M_R, R_c, M_s):
    return (M_T - np.floor(R_c * M_s) + 1) * (M_R - np.floor(R_c * M_s) + 1)

# Figure and Subplots 설정
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
axs = axs.flatten()

# 1. M_s를 고정하고 안테나 개수를 변형시키면서 R_c값 별로 diversity 비교
M_s_fixed = 2  # 고정된 M_s

for R_c in R_c_values:
    diversity_orders = [diversity_order(M, M, R_c, M_s_fixed) for M in antenna_values]
    axs[0].plot(antenna_values, diversity_orders, marker='o', label=f'R_c={R_c}')

axs[0].set_title(f'Diversity Order vs. Number of Antennas for Fixed M_s={M_s_fixed}')
axs[0].set_xlabel('Number of Antennas (M_T = M_R)')
axs[0].set_ylabel('Diversity Order (D)')
axs[0].legend()
axs[0].grid(True)

# 2. M_s를 고정하고 R_c값에 따라 안테나 개수 별로 diversity 비교 (M_T = M_R = 2, 4, 8)
M_s_fixed = 2  # 고정된 M_s
selected_antenna_values = [2, 4, 8]  # 선택된 안테나 값들

for M in selected_antenna_values:
    diversity_orders = [diversity_order(M, M, R_c, M_s_fixed) for R_c in R_c_values]
    axs[1].plot(R_c_values, diversity_orders, marker='o', label=f'M_T = M_R = {M}')

axs[1].set_title(f'Diversity Order vs. Code Rate (R_c) for Fixed M_s={M_s_fixed}')
axs[1].set_xlabel('Code Rate (R_c)')
axs[1].set_ylabel('Diversity Order (D)')
axs[1].legend()
axs[1].grid(True)

# 3. R_c를 고정하고, 안테나 개수에 따른 M_s값 별로 diversity 비교
R_c_fixed = 1  # 고정된 R_c

for M_s in M_s_values:
    diversity_orders = [diversity_order(M, M, R_c_fixed, M_s) for M in antenna_values]
    axs[2].plot(antenna_values, diversity_orders, marker='o', label=f'M_s={M_s}')

axs[2].set_title(f'Diversity Order vs. Number of Antennas for Fixed R_c={R_c_fixed}')
axs[2].set_xlabel('Number of Antennas (M_T = M_R)')
axs[2].set_ylabel('Diversity Order (D)')
axs[2].legend()
axs[2].grid(True)

# 4. R_c를 고정하고, M_s에 따른 안테나 개수 별로 diversity 비교 (M_T = M_R = 2, 4, 8)
R_c_fixed = 1  # 고정된 R_c
selected_antenna_values = [2, 4, 8]  # 선택된 안테나 값들

for M in selected_antenna_values:
    diversity_orders = [diversity_order(M, M, R_c_fixed, M_s) for M_s in M_s_values]
    axs[3].plot(M_s_values, diversity_orders, marker='o', label=f'M_T = M_R = {M}')

axs[3].set_title(f'Diversity Order vs. Spatial Multiplexing Level (M_s) for Fixed R_c={R_c_fixed}')
axs[3].set_xlabel('Spatial Multiplexing Level (M_s)')
axs[3].set_ylabel('Diversity Order (D)')
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()
