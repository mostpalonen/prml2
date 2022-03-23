import matplotlib.pyplot as plt
import numpy as np


def main():
    # Create signals (clean and noisy)
    n = np.arange(500, 600)
    y = np.concatenate((np.zeros(500), np.cos(2 * np.pi * 0.1 * n), np.zeros(300)))
    y_n = y + np.sqrt(0.5) * np.random.randn(y.size)

    det = np.convolve(y_n, np.flip(y), 'same')

    plot(y, y_n, det)


def plot(y, y_n, d):
    fig, axs = plt.subplots(3, 1)

    axs[0].plot(y)
    axs[0].set_title('Noiseless Signal')

    axs[1].plot(y_n)
    axs[1].set_title('Noisy Signal')

    axs[2].plot(d)
    axs[2].set_title('Detection Result')

    plt.show()


if __name__ == '__main__':
    main()