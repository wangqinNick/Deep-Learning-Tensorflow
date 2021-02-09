from matplotlib import pyplot as plt

temperature = [15, 13, 14.5, 17, 20, 25, 26, 26, 27, 22, 18, 15]
time = range(2, 26, 2)
plt.figure(figsize=(20, 8), dpi=80)

# Set x-axis ticks
plt.xticks(time)
plt.yticks(range(0, 30))

plt.plot(time, temperature)
# plt.savefig("./f1.svg")
plt.show()
