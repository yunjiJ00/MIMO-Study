import numpy as np
import matplotlib.pyplot as plt

def channel_capacity(snr_db):
    """
    Calculate channel capacity given SNR in dB.
    
    Parameters:
    snr_db (float): Signal-to-noise ratio in dB.
    
    Returns:
    float: Channel capacity in bits per second per Hz.
    """
    snr_linear = 10 ** (snr_db / 10)
    return np.log2(1 + snr_linear)

def simulate_capacities(snr_db_range, tx_antennas):
    """
    Simulate channel capacities for horizontal and vertical encoding.
    
    Parameters:
    snr_db_range (np.ndarray): Array of SNR values in dB.
    tx_antennas (int): Number of transmit antennas.
    
    Returns:
    tuple: (horizontal_capacities, vertical_capacities)
    """
    horizontal_capacities = []
    vertical_capacities = []
    
    for snr_db in snr_db_range:
        # Horizontal Encoding: Capacity per antenna (independent)
        horizontal_capacity = channel_capacity(snr_db)
        
        # Vertical Encoding: Total capacity considering multiple antennas
        vertical_capacity = tx_antennas * channel_capacity(snr_db)
        
        horizontal_capacities.append(horizontal_capacity)
        vertical_capacities.append(vertical_capacity)
    
    return np.array(horizontal_capacities), np.array(vertical_capacities)

# Parameters
tx_antennas = 4  # Number of transmit antennas
snr_db_range = np.linspace(-5, 31, 100)  # SNR range from 0 to 20 dB

# Simulate capacities
horizontal_capacities, vertical_capacities = simulate_capacities(snr_db_range, tx_antennas)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(snr_db_range, horizontal_capacities, label='Horizontal Encoding', color='blue')
plt.plot(snr_db_range, vertical_capacities, label='Vertical Encoding', color='green')
plt.xlabel('SNR (dB)')
plt.ylabel('Channel Capacity (bits/Hz)')
plt.title('Channel Capacity Comparison: \nHorizontal vs Vertical Encoding')
plt.legend()
plt.grid(True)
plt.show()
