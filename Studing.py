import numpy as np
import matplotlib.pyplot as plt

# 파라미터 설정
M_T_values = [2, 4, 8]  # 송신 안테나 수
M_R_values = [2, 4, 8]  # 수신 안테나 수
R_c_values = [0.5, 1, 1.5]  # 코드율 값
M_s_values = np.arange(1, 5)  # Spatial Multiplexing 레벨

def diversity_order(M_T, M_R, R_c, M_s):
    return (M_T - np.floor(R_c * M_s) + 1) * (M_R - np.floor(R_c * M_s) + 1)

def uncoded_diversity_order(M_T, M_R, M_s):
    return (M_T - M_s + 1) * (M_R - M_s + 1)

def beamforming_diversity_order(M_T, M_R):
    return M_T * M_R

def full_diversity_order():
    return 1

# Figure and Subplots 설정
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
axs = axs.flatten()

# 첫 번째 서브플롯: R_c가 일정할 때, 안테나 수에 따른 Diversity Order
R_c_fixed = 1  # 예를 들어, 코드율을 1로 고정

for M_T in M_T_values:
    for M_R in M_R_values:
        diversity_orders = [diversity_order(M_T, M_R, R_c_fixed, M_s) for M_s in M_s_values]
        axs[0].plot(M_s_values, diversity_orders, marker='o', label=f'M_T={M_T}, M_R={M_R}')

axs[0].set_title('Diversity Order for Fixed R_c (R_c=1)')
axs[0].set_xlabel('Spatial Multiplexing Level (M_s)')
axs[0].set_ylabel('Diversity Order (D)')
axs[0].legend()
axs[0].grid(True)

# 두 번째 서브플롯: 안테나 수가 일정할 때, R_c에 따른 Diversity Order
M_T_fixed = 4  # 예를 들어, 송신 안테나 수를 4로 고정
M_R_fixed = 4  # 예를 들어, 수신 안테나 수를 4로 고정

for R_c in R_c_values:
    diversity_orders = [diversity_order(M_T_fixed, M_R_fixed, R_c, M_s) for M_s in M_s_values]
    axs[1].plot(M_s_values, diversity_orders, marker='o', label=f'R_c={R_c}')

axs[1].set_title('Diversity Order for Fixed M_T and M_R (M_T=4, M_R=4)')
axs[1].set_xlabel('Spatial Multiplexing Level (M_s)')
axs[1].set_ylabel('Diversity Order (D)')
axs[1].legend()
axs[1].grid(True)

# 세 번째 서브플롯: Uncoded 시스템의 경우
for M_T in M_T_values:
    for M_R in M_R_values:
        diversity_orders = [uncoded_diversity_order(M_T, M_R, M_s) for M_s in M_s_values]
        axs[2].plot(M_s_values, diversity_orders, marker='o', label=f'M_T={M_T}, M_R={M_R}')

axs[2].set_title('Diversity Order for Uncoded System')
axs[2].set_xlabel('Spatial Multiplexing Level (M_s)')
axs[2].set_ylabel('Diversity Order (D)')
axs[2].legend()
axs[2].grid(True)

# 네 번째 서브플롯: Beamforming 및 Full Diversity
for M_T in M_T_values:
    diversity_orders_beamforming = [beamforming_diversity_order(M_T, M_T) for _ in M_s_values]
    axs[3].plot(M_s_values, diversity_orders_beamforming, marker='o', label=f'M_T={M_T} (Beamforming)')

    diversity_orders_full = [full_diversity_order()] * len(M_s_values)
    axs[3].plot(M_s_values, diversity_orders_full, linestyle='--', label=f'M_T={M_T} (Full Diversity)')

axs[3].set_title('Diversity Order for Beamforming and Full Diversity')
axs[3].set_xlabel('Spatial Multiplexing Level (M_s)')
axs[3].set_ylabel('Diversity Order (D)')
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()
