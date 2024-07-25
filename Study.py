import numpy as np
import matplotlib.pyplot as plt

def generate_bpsk_symbols(num_bits):
    return 2 * np.random.randint(0, 2, num_bits) - 1

def apply_mimo_channel(symbols, num_tx, num_rx, snr_db):
    snr_linear = 10**(snr_db / 10)
    noise_variance = 1 / snr_linear
    h = np.random.normal(0, 1, (num_rx, num_tx)) + 1j * np.random.normal(0, 1, (num_rx, num_tx))
    noise = np.sqrt(noise_variance/2) * (np.random.normal(0, 1, (num_rx, symbols.shape[1])) + 1j * np.random.normal(0, 1, (num_rx, symbols.shape[1])))
    received_signal = h @ symbols + noise
    return received_signal, h

def ml_detection(received_signal, h):
    num_rx, num_tx = h.shape
    num_symbols = received_signal.shape[1]
    possible_symbols = np.array(np.meshgrid(*[[-1, 1]] * num_tx)).T.reshape(-1, num_tx)
    detected_symbols = np.zeros((num_tx, num_symbols), dtype=int)
    
    for i in range(num_symbols):
        min_distance = np.inf
        best_symbol = None
        for symbol in possible_symbols:
            distance = np.linalg.norm(received_signal[:, i] - h @ symbol)
            if distance < min_distance:
                min_distance = distance
                best_symbol = symbol
        detected_symbols[:, i] = best_symbol
    print('ML detection completed for a channel realization.')
    return detected_symbols

def calculate_ber(num_bits, num_tx, num_rx, snr_db, num_channels, num_symbols):
    total_bit_errors = 0
    total_bits_sent = num_bits * num_channels * num_symbols

    for i in range(num_channels):
        symbols = generate_bpsk_symbols(num_bits * num_symbols)
        transmitted_symbols = symbols.reshape((num_tx, -1))
        received_signal, h = apply_mimo_channel(transmitted_symbols, num_tx, num_rx, snr_db)
        detected_bits = ml_detection(received_signal, h).reshape(-1)
        bit_errors = np.sum(detected_bits != (symbols > 0))
        total_bit_errors += bit_errors
        
        if (i + 1) % 10 == 0:  # 10개 채널 처리 후 진행 상황을 출력
            print(f"Processed {i + 1}/{num_channels} channels.")

    ber = total_bit_errors / total_bits_sent
    return ber

def simulate_ber(num_bits, num_tx, num_rx, snr_range, num_channels, num_symbols):
    ber_results = []
    total_snr_values = len(snr_range)
    for idx, snr_db in enumerate(snr_range):
        print(f"Processing SNR {snr_db} dB ({idx + 1}/{total_snr_values})")
        ber = calculate_ber(num_bits, num_tx, num_rx, snr_db, num_channels, num_symbols)
        ber_results.append(ber)
        print(f"SNR {snr_db} dB completed with BER: {ber:.6f}")
    return ber_results

# Simulation parameters
num_bits = 100  # Number of bits per transmission
snr_range = range(-11, 31, 2)  # SNR range from -11 to 31 dB
num_channels = 100  # Number of different channel realizations
num_symbols = 1000  # Number of symbols per channel realization

# Plotting the results
plt.figure(figsize=(10, 6))
plt.semilogy(snr_range, simulate_ber(num_bits, 4, 4, snr_range, num_channels, num_symbols), 'o-', label="4x4 MIMO")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("BER vs SNR for MIMO BPSK System with ML Detection")
plt.grid(True)
plt.legend()
plt.show()
