import numpy as np
import matplotlib.pyplot as plt

class MIMOSystem:
    def __init__(self, num_transmit_antennas=2, num_receive_antennas=3, num_symbols=1000, noise_power=0.1):
        self.num_transmit_antennas = num_transmit_antennas
        self.num_receive_antennas = num_receive_antennas
        self.num_symbols = num_symbols
        self.noise_power = noise_power
        
        # 채널 게인 초기화 (가우시안 노이즈 추가)
        self.channel_gain = np.random.randn(num_receive_antennas) + 1j * np.random.randn(num_receive_antennas)
        
        # 수신된 신호 및 보낸 신호 초기화
        self.transmitted_signal = None
        self.received_signals = np.zeros((num_receive_antennas, num_symbols), dtype=complex)

    def generate_transmitted_symbols(self):
        # 전송할 가우시안 분포의 심볼 생성 (여기서는 단순화를 위해 하나의 신호만 생성)
        self.transmitted_signal = np.random.randn(self.num_symbols) + 1j * np.random.randn(self.num_symbols)

    def simulate_channel(self):
        # 노이즈 생성
        noise = np.sqrt(self.noise_power) * (np.random.randn(self.num_symbols, self.num_receive_antennas) + 1j * np.random.randn(self.num_symbols, self.num_receive_antennas))
        
        # 각 안테나에서의 수신 신호 계산
        for i in range(self.num_receive_antennas):
            self.received_signals[i] = self.channel_gain[i] * self.transmitted_signal + noise[:, i]

    def calculate_errors(self):
        # 각 수신 안테나에서의 MSE 계산
        mse = np.zeros(self.num_receive_antennas)
        for i in range(self.num_receive_antennas):
            mse[i] = np.mean(np.abs(self.received_signals[i] - self.transmitted_signal) ** 2)
        return mse
    
    def plot_transmit_receive_comparison(self):
        error = self.calculate_errors()
        best_antenna = np.argmin(error) + 1  # error가 가장 작은 안테나

        # 보낸 신호 및 각 수신 안테나에서의 수신된 데이터 비교 플롯
        plt.figure(figsize=(18, 18))
        
        for i in range(self.num_receive_antennas):
            # 실수부 플롯
            plt.subplot(self.num_receive_antennas, 2, 2 * i + 1)
            plt.plot(np.real(self.transmitted_signal), label='Transmitted Signal (Real)', alpha=0.6)
            plt.plot(np.real(self.received_signals[i]), label=f'Received Signal at Antenna {i + 1} (Real)')
            plt.title(f'Transmitted Signal vs. Received Signal at Antenna {i + 1} (Real Part)')
            plt.xlabel('Symbol Index')
            plt.ylabel('Amplitude')
            plt.legend()
            
            # 허수부 플롯
            plt.subplot(self.num_receive_antennas, 2, 2 * i + 2)
            plt.plot(np.imag(self.transmitted_signal), label='Transmitted Signal (Imag)', alpha=0.6)
            plt.plot(np.imag(self.received_signals[i]), label=f'Received Signal at Antenna {i + 1} (Imag)')
            plt.title(f'Transmitted Signal vs. Received Signal at Antenna {i + 1} (Imaginary Part)')
            plt.xlabel('Symbol Index')
            plt.ylabel('Amplitude')
            plt.legend()

        plt.suptitle(f'Best Antenna: Antenna {best_antenna} with Mean Error = {error[best_antenna - 1]:.4f}', fontsize=16)
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def plot_error_comparison(self):
        error = self.calculate_errors()
        antennas = np.arange(1, self.num_receive_antennas + 1)

        plt.figure(figsize=(10, 6))
        plt.bar(antennas, error, color='skyblue')
        plt.xlabel('Antenna')
        plt.ylabel('Mean Error')
        plt.title('Mean Error of Received Signals at Different Antennas')
        plt.xticks(antennas)
        plt.show()

    def plot_combined_signal_comparison(self):
        # 모든 수신 신호를 합해서 원래 신호를 추정
        combined_signal = np.sum(self.received_signals, axis=0) / self.num_receive_antennas
        
        # 원래 신호와 추정된 신호의 비교 플롯
        plt.figure(figsize=(18, 6))

        # 실수부 플롯
        plt.subplot(1, 2, 1)
        plt.plot(np.real(self.transmitted_signal), label='Transmitted Signal (Real)', alpha=0.6)
        plt.plot(np.real(combined_signal), label='Combined Received Signal (Real)', linestyle='--')
        plt.title('Transmitted Signal vs. Combined Received Signal (Real Part)')
        plt.xlabel('Symbol Index')
        plt.ylabel('Amplitude')
        plt.legend()

        # 허수부 플롯
        plt.subplot(1, 2, 2)
        plt.plot(np.imag(self.transmitted_signal), label='Transmitted Signal (Imag)', alpha=0.6)
        plt.plot(np.imag(combined_signal), label='Combined Received Signal (Imag)', linestyle='--')
        plt.title('Transmitted Signal vs. Combined Received Signal (Imaginary Part)')
        plt.xlabel('Symbol Index')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.show()

# 예제 실행
if __name__ == '__main__':
    num_transmit_antennas=2
    num_receive_antennas=3
    num_symbols=300
    noise_power=0.1
    mimo_sim = MIMOSystem(num_transmit_antennas, num_receive_antennas, num_symbols, noise_power)
    mimo_sim.generate_transmitted_symbols()
    mimo_sim.simulate_channel()
    
    # 보낸 신호와 각 수신 안테나에서의 수신된 신호 비교
    mimo_sim.plot_transmit_receive_comparison()
    
    # 각 수신 안테나에서의 MSE 비교
    mimo_sim.plot_error_comparison()
    
    # 모든 수신 신호를 합성한 신호와 원래 신호 비교
    mimo_sim.plot_combined_signal_comparison()
