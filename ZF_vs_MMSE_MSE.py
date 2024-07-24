import numpy as np
import matplotlib.pyplot as plt

# 파라미터 설정
num_symbols = 4   # 송신 심볼의 수
num_antennas = 4  # 안테나의 수

# 잡음의 분산 설정
sigma_n = 1  # 잡음의 분산

# SNR 범위 설정 (dB로 설정 후 선형 스케일로 변환)
SNR_dB = np.arange(-5, 35, 2)  # SNR 범위 (-5dB, -3dB, ..., 33dB)
SNR_linear = 10 ** (SNR_dB / 10)
alpha = 1 / SNR_linear  # MMSE에서 사용할 alpha

# ZF와 MMSE의 MSE를 저장할 배열
mse_zf = []
mse_mmse = []
# 채널 행렬 생성
H = np.random.randn(num_antennas, num_symbols) + 1j * np.random.randn(num_antennas, num_symbols)
H = H / np.sqrt(2)  # 채널 행렬 정규화

# 송신 심볼 생성 (BPSK)
x = np.random.choice([-1, 1], num_symbols)  # 송신 심볼

for snr, alpha_val in zip(SNR_linear, alpha):
    # 수신 신호 생성
    noise = (np.random.randn(num_antennas) + 1j * np.random.randn(num_antennas)) * np.sqrt(sigma_n / 2)
    y = np.dot(H, x) + noise / np.sqrt(snr)  # SNR에 따라 노이즈 조절

    # Zero Forcing (ZF) 공분산 행렬
    H_H_H_H_inv = np.linalg.inv(H.conj().T @ H)
    covariance_zf = sigma_n**2 * H_H_H_H_inv
    mse_zf.append(np.mean(np.diag(covariance_zf)))

    # Minimum Mean Square Error (MMSE) 공분산 행렬
    covariance_mmse = sigma_n**2 * np.linalg.inv(H.conj().T @ H + alpha_val * np.eye(num_symbols))
    mse_mmse.append(np.mean(np.diag(covariance_mmse)))

# 플롯 생성
plt.figure(figsize=(10, 6))
plt.plot(SNR_dB, mse_zf, 'o-', label='ZF MSE')
plt.plot(SNR_dB, mse_mmse, 's-', label='MMSE MSE')
plt.xlabel('SNR (dB)')
plt.ylabel('Mean Square Error (MSE)')
plt.title('ZF vs MMSE MSE at Different SNR')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
