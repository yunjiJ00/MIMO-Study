import os
import io
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

class CapacityCalculater:
    def __init__(self, MT=2, MR=2, N0=1, Es=1):
        self.SNR_dB = np.linspace(-5, 30, 100)  # SNR range from 0 to 30 dB
        self.SNR = 10**(self.SNR_dB / 10)  # Convert SNR from dB to linear scale
        self.N0 = N0  # Noise power
        self.Es = Es  # Signal power
        self.MT = MT
        self.MR = MR
    
    # Function to calculate MIMO capacity
    def mimo_capacity(self, channel_num, SNR, MT, MR):
        sum_capacity = 0
        for _ in range(channel_num):
            H = np.random.randn(MR, MT) + 1j * np.random.randn(MR, MT)  # Random Gaussian channel matrix
            Rss = self.Es * np.eye(MT)  # Signal covariance matrix
            Rn = self.N0 * np.eye(MR)  # Noise covariance matrix
            capacity = np.log2(np.linalg.det(np.eye(MR) + (SNR / MT) * H @ Rss @ H.conj().T))
            sum_capacity += capacity
        final_capacity = sum_capacity/channel_num
        return final_capacity

    # Function to calculate SIMO capacity
    def simo_capacity(self, channel_num, SNR, MR):
        sum_capacity = 0
        for _ in range(channel_num):
            h = np.random.randn(MR) + 1j * np.random.randn(MR)  # Random Gaussian channel vector
            capacity = np.log2(1 + SNR * np.sum(np.abs(h)**2))
            sum_capacity += capacity
        final_capacity = sum_capacity/channel_num
        return final_capacity

    # Function to calculate MISO capacity
    def miso_capacity(self, channel_num, SNR, MT):
        sum_capacity = 0
        for _ in range(channel_num):
            h = np.random.randn(MT) + 1j * np.random.randn(MT)  # Random Gaussian channel vector
            capacity = np.log2(1 + (SNR / MT) * np.sum(np.abs(h)**2))
            sum_capacity += capacity
        final_capacity = sum_capacity/channel_num
        return final_capacity
    
    def siso_capacity(self, channel_num, SNR):
        sum_capacity = 0
        for _ in range(channel_num):
            h = np.random.randn(1) + 1j * np.random.randn(1)  # Random Gaussian channel vector
            capacity = np.log2(1+SNR*(np.abs(h)**2))
            sum_capacity += capacity
        final_capacity = sum_capacity/channel_num
        return final_capacity
    
    def plot_capacity_vs_SNR(self):
        channel_num = 100
        mimo_capacities = [self.mimo_capacity(channel_num, snr, self.MT, self.MR) for snr in self.SNR]
        simo_capacities = [self.simo_capacity(channel_num,snr, self.MR) for snr in self.SNR]
        miso_capacities = [self.miso_capacity(channel_num,snr, self.MT) for snr in self.SNR]
        siso_capacities = [self.siso_capacity(channel_num,snr) for snr in self.SNR]
        # Plot capacities vs SNR
        plt.figure(figsize=(10, 6))
        plt.plot(self.SNR_dB, mimo_capacities, color='red', label='MIMO (MT=2, MR=2)')
        plt.plot(self.SNR_dB, simo_capacities, color='blue', label='SIMO (MR=2)')
        plt.plot(self.SNR_dB, miso_capacities, color='green', label='MISO (MT=2)')
        plt.plot(self.SNR_dB, siso_capacities, color='black', label='SISO')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Capacity (bits/s/Hz)')
        plt.title('Capacity vs SNR for MIMO, SIMO, MISO and SISO')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_capacity_vs_antennas(self):
        channel_num = 100
        num_antennas = range(1, 10)
        fixed_SNR_dB = 10  # Fixed SNR value for comparison (in dB)
        fixed_SNR = 10**(fixed_SNR_dB / 10)  # Convert to linear scale
        mimo_capacities_antennas = [self.mimo_capacity(channel_num,fixed_SNR, n, n) for n in num_antennas]
        simo_capacities_antennas = [self.simo_capacity(channel_num,fixed_SNR, n) for n in num_antennas]
        miso_capacities_antennas = [self.miso_capacity(channel_num,fixed_SNR, n) for n in num_antennas]
        # Plot capacities vs number of antennas
        plt.figure(figsize=(10, 6))
        plt.plot(num_antennas, mimo_capacities_antennas, marker='o', label='MIMO')
        plt.plot(num_antennas, simo_capacities_antennas, marker='o', label='SIMO')
        plt.plot(num_antennas, miso_capacities_antennas, marker='o', label='MISO')
        plt.xlabel('Number of Antennas')
        plt.ylabel('Capacity (bits/s/Hz)')
        plt.title('Capacity vs Number of Antennas for MIMO, SIMO, and MISO at SNR=10 dB')
        plt.xticks(num_antennas)  # Ensure x-axis uses discrete values for number of antennas
        plt.legend()
        plt.grid()
        plt.show()
        
    def plot_mimo_capacity(self):
        channel_num = 100
        siso_capacities = [self.siso_capacity(channel_num, snr) for snr in self.SNR]
        mimo_capacities_3s = [self.mimo_capacity(channel_num,snr, 3, 3) for snr in self.SNR]
        mimo_capacities_4s = [self.mimo_capacity(channel_num,snr, 4, 4) for snr in self.SNR]
        mimo_capacities_6s = [self.mimo_capacity(channel_num,snr, 6, 6) for snr in self.SNR]
        mimo_capacities_8s = [self.mimo_capacity(channel_num,snr, 8, 8) for snr in self.SNR]
        # Plot capacities vs number of antennas
        plt.figure(figsize=(10, 6))
        plt.plot(self.SNR_dB, siso_capacities, color='skyblue', marker='.', label='1x1 SISO')
        plt.plot(self.SNR_dB, mimo_capacities_3s, color='black', marker='.', label='3x3 MIMO')
        plt.plot(self.SNR_dB, mimo_capacities_4s, color='green', marker='.', label='4x4 MIMO')
        plt.plot(self.SNR_dB, mimo_capacities_6s, color='blue', marker='.', label='6x6 MIMO')
        plt.plot(self.SNR_dB, mimo_capacities_8s, color='red', marker='.', label='8x8 MIMO')
        plt.xlabel('SNR(dB)')
        plt.ylabel('Capacity (bits/s/Hz)')
        plt.title('Capacity vs Number of Antennas for MIMO at SNR range (0, 30)')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_capacity_vs_channel_num(self, gif_filename='capacity_vs_channel_num.gif'):
        channel = range(1, 1000)  # Adjust range for testing
        images = []
        
        for i in channel:
            mimo_capacities = [self.mimo_capacity(i, snr, self.MT, self.MR) for snr in self.SNR]
            simo_capacities = [self.simo_capacity(i, snr, self.MR) for snr in self.SNR]
            miso_capacities = [self.miso_capacity(i, snr, self.MT) for snr in self.SNR]
            siso_capacities = [self.siso_capacity(i, snr) for snr in self.SNR]
            
            # Plot capacities vs SNR
            plt.figure(figsize=(10, 6))
            plt.plot(self.SNR_dB, mimo_capacities, color='red', label='MIMO (MT=2, MR=2)')
            plt.plot(self.SNR_dB, simo_capacities, color='blue', label='SIMO (MR=2)')
            plt.plot(self.SNR_dB, miso_capacities, color='green', label='MISO (MT=2)')
            plt.plot(self.SNR_dB, siso_capacities, color='black', label='SISO')
            plt.xlabel('SNR (dB)')
            plt.ylabel('Capacity (bits/s/Hz)')
            plt.title(f'Capacity vs SNR for MIMO, SIMO, MISO and SISO (Channel {i})')
            plt.legend()
            plt.grid()
            
            # Save plot to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            images.append(imageio.imread(buf))
            print("processing", i)
        
        # Create GIF from images in memory
        imageio.mimsave(gif_filename, images, duration=0.1)

def main():
    # Calculate capacities for various numbers of antennas
    capacity = CapacityCalculater(5, 5)
    capacity.plot_capacity_vs_SNR()
    capacity.plot_capacity_vs_antennas()
    capacity.plot_mimo_capacity()
    capacity.plot_capacity_vs_channel_num()

if __name__ == "__main__":
    main()