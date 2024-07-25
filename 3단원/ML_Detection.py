import numpy as np
import itertools
import matplotlib.pyplot as plt

# 예시 데이터 설정
N = 2  # 변조 수준 (BPSK: -1, 1)
M_T = 3  # 송신 심볼의 수
# 계산 복잡도 플롯
N_values = [2, 4, 8, 16, 32, 64]
M_T_values = range(1, 15)

H = np.random.randn(3, 3)  # 임의의 3x3 채널 행렬
y = np.random.randn(3)  # 임의의 3차원 수신 벡터

# 가능한 모든 송신 벡터 생성 (BPSK: -1, 1)
possible_symbols = [-1, 1]
all_possible_x = list(itertools.product(possible_symbols, repeat=M_T))

# 각 송신 벡터에 대한 유클리드 거리 계산
distances = []
for x in all_possible_x:
    x = np.array(x)
    distance = np.linalg.norm(y - np.dot(H, x))**2
    distances.append((x, distance))

# 유클리드 거리를 기준으로 정렬
distances.sort(key=lambda x: x[1])

best_x = distances[0][0]
min_distance = distances[0][1]

# 정렬된 리스트에서 x_labels와 y_values 생성
sorted_x_labels = [''.join(map(str, x[0])) for x in distances]
sorted_y_values = [x[1] for x in distances]

# 원래 순서로 복원된 리스트 생성
original_x_labels = [''.join(map(str, x)) for x in all_possible_x]
original_y_values = []
for x in all_possible_x:
    x = np.array(x)
    distance = np.linalg.norm(y - np.dot(H, x))**2
    original_y_values.append(distance)

plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
bars = plt.bar(original_x_labels, original_y_values, color='blue')
plt.xlabel('Possible transmitted vectors')
plt.ylabel('Euclidean distance')
plt.title('Euclidean distance for each possible transmitted vector')
plt.xticks(rotation=90)
plt.grid(True)

# 바 위에 텍스트 값 추가
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01 * max(original_y_values), f'{yval:.2f}', ha='center', va='bottom')

# 텍스트를 x축 레이블 아래에 위치시키기
plt.text(len(original_x_labels) / 2, -0.2 * max(original_y_values), f'Optimal transmission vector: {best_x}', ha='center')
plt.text(len(original_x_labels) / 2, -0.23 * max(original_y_values), f'Minimum Euclidean distance: {min_distance}', ha='center')

# 총 후보 수와 계산 복잡도를 저장할 리스트
total_candidates = []
computational_complexity = []

# 총 후보 수와 계산 복잡도를 계산
for N in N_values:
    candidates_for_N = []
    complexity_for_N = []
    for M_T in M_T_values:
        candidates = N ** M_T
        complexity = M_T * np.log2(N) # 심볼당 비트 수는 log2(N)
        candidates_for_N.append(candidates)
        complexity_for_N.append(complexity)
    total_candidates.append(candidates_for_N)
    computational_complexity.append(complexity_for_N)

plt.subplot(1, 2, 2)
for i, N in enumerate(N_values):
    plt.plot(M_T_values, computational_complexity[i], marker='o', linestyle='-', label=f'N={N}')
plt.xlabel('Number of transmitted symbols (M_T)')
plt.ylabel('Computational Complexity')
plt.title('Computational Complexity vs. M_T')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

