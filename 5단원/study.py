import numpy as np
import matplotlib.pyplot as plt

# 점대점 채널 용량 계산 함수
def point_to_point_capacity(N):
    SNR = 1  # 임의의 SNR 값
    return N * np.log2(1 + SNR)

# 다중 사용자 시스템의 합용량 계산 함수
def multiuser_capacity(K, M):
    min_KM = min(K, M)
    SNR = 1  # 임의의 SNR 값
    return min_KM * np.log2(1 + SNR)

# 송신 및 수신 안테나 수 정의
N_values = [2, 4, 8, 16]

# 점대점 채널 용량 데이터
point_to_point_capacities = [point_to_point_capacity(N) for N in N_values]

# 다중 사용자 시스템의 합용량 데이터
multiuser_capacities = [multiuser_capacity(N, N) for N in N_values]

# 그래프 생성
plt.figure(figsize=(14, 6))

# 점대점 채널 용량 그래프
plt.subplot(1, 2, 1)
plt.plot(N_values, point_to_point_capacities, marker='o', linestyle='-', color='b', label='Point-to-Point Capacity')
plt.xlabel('Number of Antennas (K = M)')
plt.ylabel('Capacity')
plt.title('Point-to-Point Capacity vs. Number of Antennas')
plt.legend()
plt.grid(True)

# 다중 사용자 시스템의 합용량 그래프
plt.subplot(1, 2, 2)
plt.plot(N_values, multiuser_capacities, marker='o', linestyle='-', color='r', label='Multiuser Capacity')
plt.xlabel('Number of Antennas (K = M)')
plt.ylabel('Total Capacity')
plt.title('Multiuser Capacity vs. Number of Antennas')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
