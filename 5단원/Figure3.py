import numpy as np
import matplotlib.pyplot as plt

# SNR 범위 설정 (dB)
snr_dB = np.linspace(-10, 30, 10)  # 더 많은 점을 사용하여 더욱 부드러운 곡선 생성
snr = 10 ** (snr_dB / 10)

def average_prob(K, M, rho, alpha=1):
    """Regularized Inversion의 평균 확률을 계산하는 함수"""
    P_avg_reg = np.zeros_like(rho)
    for i in range(len(rho)):
        P_avg_reg[i] = (1 / M) * np.sum([1 / (1 + rho[i] / (k + alpha)) for k in range(1, K + 1)])
    return P_avg_reg

def average_prob_reqularized(K, M, rho):
    """Regularized Inversion의 평균 확률을 계산하는 함수"""
    P_avg_reg = np.zeros_like(rho)
    for i in range(len(rho)):
        if rho[i] == 0:  # rho가 0인 경우
            alpha = 0
        elif rho[i] > 0:  # rho가 양수일 때
            alpha = K/rho[i]
        else:  # rho가 음수일 때
            alpha = - K/rho[i]
        P_avg_reg[i] = (1 / M) * np.sum([1 / (1 + rho[i] / (k + alpha)) for k in range(1, K + 1)])
    return P_avg_reg

# 오류 확률 계산
prob_err_4_channel = average_prob(4, 4, snr)
prob_err_4_regularized = average_prob_reqularized(4, 4, snr)
prob_err_10_channel = average_prob(10, 10, snr)
prob_err_10_regularized = average_prob_reqularized(10, 10, snr)

# 그래프 그리기
plt.figure(figsize=(12, 8))

# 10x10 QPSK 그래프
plt.semilogy(snr_dB, prob_err_10_channel, label='10x10 Channel Inversion', marker='X', linestyle='--')
plt.semilogy(snr_dB, prob_err_10_regularized, label='10x10 Regularized Inversion', marker='D', linestyle='--')
# 4x4 QPSK 그래프
plt.semilogy(snr_dB, prob_err_4_channel, label='4x4 Channel Inversion', marker='*', linestyle='-')
plt.semilogy(snr_dB, prob_err_4_regularized, label='4x4 Regularized Inversion', marker='d', linestyle='-')

# 그래프 설정
plt.title('Results for M=K=4 and M=K=10, QPSK')
plt.xlabel('SNR (ρ in dB)')
plt.ylabel('Average Probability of Error (SEP)')
plt.legend()
plt.grid(True)

# 그래프 표시
plt.show()
