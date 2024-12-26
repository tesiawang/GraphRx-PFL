import matplotlib.pyplot as plt

num_bs_ant = [2, 4, 8, 16, 32, 64, 128]
avg_SIR_1 = [7.12, 7.28, 7.98, 9.37, 11.69, 14.71, 18.72]
avg_SIR_2 = [3.21, 3.36, 3.99, 5.17, 7.59, 10.53, 13.65]
avg_SIR_3 = [1.08, 1.23, 1.85, 3.01, 5.22, 8.09, 11.04]

fig = plt.figure()
plt.plot(num_bs_ant, avg_SIR_1, label='One interferer', marker='o',markerfacecolor='none')
plt.plot(num_bs_ant, avg_SIR_2, label='Two interferers', marker='d',markerfacecolor='none')
plt.plot(num_bs_ant, avg_SIR_3, label='Three interferers', marker='s',markerfacecolor='none')
plt.xticks([2, 4, 8, 16, 32, 64, 128])

plt.xlabel('Number of BS antennas')
plt.ylabel('Average SIR (dB)')
plt.title('Average SIR vs Number of BS antennas')


plt.legend()
plt.savefig('avg_SIR_vs_num_bs_ant.png')