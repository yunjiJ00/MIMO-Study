import numpy as np
import matplotlib.pyplot as plt

# SNR 범위 설정 (dB)
SNR_dB = np.arange(-11, 12, 2)
SNR_linear = 10 ** (SNR_dB / 10)  # SNR을 선형 스케일로 변환

def pep(SNR_linear, lambda_val, M_R):
    """ PEP 계산 함수 """
    return ((lambda_val * (SNR_linear / 4)) + 1) ** (-M_R)

def ber_from_pep(pep_vals, num_symbols):
    """ PEP로부터 BER 계산 함수 """
    return pep_vals * num_symbols

# 시뮬레이션 파라미터
lambda_val = 1  # λ는 일반적으로 1로 가정할 수 있습니다.
M_R_values = [1, 4, 16, 64]  # 1x1, 2x2, 4x4 MIMO 시스템에 해당하는 수신 안테나 수
num_symbols = 1000000000  # 심볼의 수를 1000으로 설정

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
# 각 M_R에 대해 BER을 계산하고 플롯
for M_R in M_R_values:
    pep_vals = pep(SNR_linear, lambda_val, M_R)
    ber_vals = ber_from_pep(pep_vals, num_symbols)  # PEP로부터 BER 계산
    plt.plot(SNR_dB, ber_vals, marker='.', linestyle='-', label=f'{int(np.sqrt(M_R))}x{int(np.sqrt(M_R))} MIMO')
plt.yscale('log')  # BER은 종종 로그 스케일로 표시
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER vs SNR for Different MIMO Configurations')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)

# 각 M_R에 대해 BER을 계산하고 플롯
plt.subplot(1, 2, 2)
for M_R in M_R_values:
    pep_vals = pep(SNR_linear, lambda_val, M_R)
    plt.plot(SNR_dB, pep_vals, marker='.', linestyle='-', label=f'{int(np.sqrt(M_R))}x{int(np.sqrt(M_R))} MIMO')

plt.yscale('log')  # BER은 종종 로그 스케일로 표시
plt.xlabel('SNR (dB)')
plt.ylabel('PEP')
plt.title('PEP vs SNR for Different MIMO Configurations')
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.legend()
plt.show()
