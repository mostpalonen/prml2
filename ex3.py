import matplotlib.pyplot as plt
import numpy as np


def main():
    # Create noisy signal and detectors
    n = np.arange(100)
    y = np.concatenate((np.zeros(500), np.cos(2 * np.pi * 0.1 * n), np.zeros(300)))
    exp_y = np.exp(-2*np.pi * 1j * 0.1 * n)
    y_n = y + np.sqrt(0.5) * np.random.randn(y.size)
    
    # Convolve signals with detector
    det1 = np.convolve(y, y_n, 'same')
    det1full = np.convolve(y, y_n, 'full')
    det2 = np.convolve(exp_y, y_n, 'same')

    # Plot detections
    plot(y, y_n, det1full[500:1500], det1, det2)


def plot(y, y_n, d1, d1full, d2):
    _, axs = plt.subplots(4, 1)

    axs[0].plot(y)
    axs[0].set_title('Noiseless Signal')

    axs[1].plot(y_n)
    axs[1].set_title('Noisy Signal')

    axs[2].plot(d1)
    axs[2].plot(d1full)
    axs[2].set_title('Detection with cos. Blue = manually centered full convolution, Oragne = "same" convolution')

    axs[3].plot(d2)
    axs[3].set_title('Detection with exp')

    plt.show()


if __name__ == '__main__':
    main()