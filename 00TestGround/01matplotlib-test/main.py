from matplotlib import pyplot as plt
from numpy import random


def l1():
    temperature = [15, 13, 14.5, 17, 20, 25, 26, 26, 27, 22, 18, 15]
    time = range(2, 26, 2)
    plt.figure(figsize=(20, 8), dpi=80)

    x = [i / 2 for i in range(4, 49)]
    # Set x-axis ticks
    plt.xticks(x[::2])
    plt.yticks(range(0, 30))

    plt.plot(time, temperature)
    # plt.savefig("./f1.svg")
    plt.show()


def l2():
    y = [random.randint(20, 35) for i in range(120)]
    x = range(0, 120)
    plt.figure(figsize=(20, 8), dpi=80)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    l2()
