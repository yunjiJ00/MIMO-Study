import numpy as np
import matplotlib.pyplot as plt

class WaaterFillingComparison:
    def __init__(self, channel_gains, total_power):
        self.channel_gains = np.array(channel_gains)
        self.total_power = total_power
    
    def water_filling(self):
        k = len(self.channel_gains)
        sorted_indices = np.argsort(self.channel_gains)
        N_sorted = self.channel_gains[sorted_indices]
        P = np.zeros(k)
        i = k
        
        while True:
            nu = (self.total_power + np.sum(N_sorted[:i])) / i
            for j in range(i):
                if nu - N_sorted[j] < 0:
                    P[j] = 0
                    i -= 1
                    break
                else:
                    P[j] = nu - N_sorted[j]
            if j == i - 1:
                break

        P_original_order = np.zeros(k)
        P_original_order[sorted_indices] = P
        
        C = np.zeros(k)
        for j in range(k):
            C[j] = np.log2(1 + P_original_order[j] / self.channel_gains[j])
        
        return P_original_order, C, np.sum(P_original_order), np.sum(C)

    
    def equal_power_allocation(self):
        n = len(self.channel_gains)
        P = np.ones(n) * (self.total_power / n)
        C = np.zeros(n)
        for j in range(n):
            C[j] = np.log2(1 + P[j] / self.channel_gains[j])
        return P, C, np.sum(P), np.sum(C)
    
    def get_water_filling_allocation(self):
        allocations, capacities, total_power_allocation, total_capacity = self.water_filling()
        return allocations, capacities, total_power_allocation, total_capacity
    
    def get_equal_power_allocation(self):
        allocations, capacities, total_power_allocation, total_capacity = self.equal_power_allocation()
        return allocations, capacities, total_power_allocation, total_capacity

    def plot_comparison(self):
        wf_allocation, wf_capacities, wf_total_power, wf_total_capacity = self.get_water_filling_allocation()
        ep_allocation, ep_capacities, ep_total_power, ep_total_capacity = self.get_equal_power_allocation()

        x = np.arange(len(self.channel_gains))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Water-Filling 그래프
        ax1.bar(x, self.channel_gains, color='skyblue', label='Channel Gains')
        ax1.bar(x, wf_allocation, bottom=self.channel_gains, color='orange', alpha=0.6, label='Water-Filling Allocation')
        for i, val in enumerate(wf_allocation):
            ax1.text(i, self.channel_gains[i] + val + 0.1, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
            ax1.text(i, self.channel_gains[i] - 0.3, f'C{i+1}: {wf_capacities[i]:.2f}', ha='center', va='top', fontsize=9, color='orange')
        ax1.text(0.5, -0.2, f'Total Power: {wf_total_power:.2f}, Total Capacity: {wf_total_capacity:.2f}', transform=ax1.transAxes, ha='center', va='center')
        ax1.set_xlabel('Channel Index')
        ax1.set_ylabel('Power Allocation')
        ax1.set_title('Water-Filling Power Allocation')
        ax1.legend()

        # Equal Power Allocation 그래프
        ax2.bar(x, self.channel_gains, color='skyblue', label='Channel Gains')
        ax2.bar(x, ep_allocation, bottom=self.channel_gains, color='red', alpha=0.6, label='Equal Power Allocation')
        for i, val in enumerate(ep_allocation):
            ax2.text(i, self.channel_gains[i] + val + 0.1, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
            ax2.text(i, self.channel_gains[i] - 0.3, f'C{i+1}: {ep_capacities[i]:.2f}', ha='center', va='top', fontsize=9, color='red')
        ax2.text(0.5, -0.2, f'Total Power: {ep_total_power:.2f}, Total Capacity: {ep_total_capacity:.2f}', transform=ax2.transAxes, ha='center', va='center')
        ax2.set_xlabel('Channel Index')
        ax2.set_ylabel('Power Allocation')
        ax2.set_title('Equal Power Allocation')
        ax2.legend()

        max_power = max(np.max(self.channel_gains + wf_allocation), np.max(self.channel_gains + ep_allocation))
        ax1.set_ylim(0, max_power + 1)
        ax2.set_ylim(0, max_power + 1)

        plt.tight_layout()
        plt.show()

def main():
    channel_gains = [2.5, 1.0, 3.0, 1.8, 0.5]
    total_power = 4

    comparison = WaaterFillingComparison(channel_gains, total_power)
    comparison.plot_comparison()

if __name__ == "__main__":
    main()
