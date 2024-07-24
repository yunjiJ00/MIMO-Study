import numpy as np
import matplotlib.pyplot as plt

# Zero Forcing MSE 계산 함수
def zero_forcing_mse(SNR_dB, H):
    sigma_squared = 1 / (10**(SNR_dB / 10))
    H_hermitian_H = H.T.conj() @ H
    mse = sigma_squared * np.trace(np.linalg.inv(H_hermitian_H))
    return mse

# MMSE MSE 계산 함수
def mmse_mse(SNR_dB, H):
    sigma_squared = 1 / (10**(SNR_dB / 10))
    H_hermitian_H = H.T.conj() @ H
    inverse_term = np.linalg.inv(H_hermitian_H + sigma_squared * np.eye(H.shape[1]))
    mse = sigma_squared * np.trace(inverse_term)
    return mse

# 랜덤 채널 행렬 생성 함수
def generate_random_channel(m, n):
    return np.random.randn(m, n) + 1j * np.random.randn(m, n)

# SNR 범위 설정 (dB 단위)
SNR_dB_range = np.linspace(-5, 31, 25)

# 채널 행렬 생성 (예: 2x2 크기)
H = generate_random_channel(2, 2)

# ZF와 MMSE의 MSE 계산
zf_mse_values = [zero_forcing_mse(snr, H) for snr in SNR_dB_range]
mmse_mse_values = [mmse_mse(snr, H) for snr in SNR_dB_range]

# 플롯 그리기
plt.figure(figsize=(10, 6))
plt.plot(SNR_dB_range, zf_mse_values, 'o-', label='Zero Forcing MSE')
plt.plot(SNR_dB_range, mmse_mse_values, 's-', label='MMSE MSE')
plt.xlabel('SNR (dB)')
plt.ylabel('Mean Square Error (MSE)')
plt.title('ZF and MMSE MSE')
plt.legend()
plt.grid(True)
plt.show()