import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def main():
    exercise('a)', samples=100, noise=0.0, error=1.0)
    exercise('b)', samples=400, noise=0.0, error=1.0)
    exercise('c)', samples=400, noise=2.1, error=1.0)
    exercise('d)', samples=400, noise=1.5, error=1.1)


def exercise(title, samples, noise, error):
    SSE_list, ML_list = find_f(error=error, samples=samples, noise=noise)

    # True sinusoid
    sin, sin_noise, A, phi = createSinusoid(noise=noise)

    # Get f_hat based on index and samples
    f_hat_sse = (np.argmin(SSE_list)/samples)*0.5
    f_hat_ml = (np.argmax(ML_list)/samples)*0.5
    
    if f_hat_ml == f_hat_sse:
        print(f"{title} Estimates for f in ML and SSE are the same")

    # Estimation of sinusoid using f_hat estimations
    n = np.arange(160)
    sin_hat = A*error * np.cos(2 * np.pi * f_hat_ml * n + phi*error)

    plot(title, SSE_list, ML_list, sin, sin_noise, sin_hat, f_hat_ml, noise)


def createSinusoid(noise):

    # Initialize parameters for sinusoidal
    n = np.arange(160)
    f0 = 0.06752728319488948
    sigma = 0.0 + noise
    phi = 0.6090665392794814
    A = 0.6669548209299414

    # Create sinusoid and add noise
    x0 = A * np.cos(2*np.pi*f0*n + phi)
    x = x0 + sigma * np.random.randn(x0.size)

    return x0, x, A, phi


def find_f(error, samples, noise):

    _, x, A, phi = createSinusoid(noise=noise)

    # Estimation parameters
    A_hat = A*error
    phi_hat = phi*error
    fRange = np.linspace(0, 0.5, samples)

    SSE_list = list()
    ML_list = list()
    for f in fRange:
        # compute SSE
        # compute likelihood
        # store them
        loss_SSE = 0
        p_total = 1
        for n, xn in enumerate(x):
            x_hat = A_hat * np.cos(2 *np.pi*f*n + phi_hat)
            loss_SSE += np.square(xn - x_hat)
            p_total *= stats.norm.pdf(x_hat, xn)
        SSE_list.append(loss_SSE)
        ML_list.append(p_total)
    return SSE_list, ML_list


def plot(title, SSE, ML, sin, sin_noise, sin_hat, f_hat, sigma2):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)

    axs[0, 0].plot(SSE)
    axs[0, 0].set_title('Squared Error')

    axs[0, 1].plot(ML)
    axs[0, 1].set_title('Likelihood')

    axs[1, 0].plot(sin)
    axs[1, 0].plot(sin_noise, 'go')
    axs[1, 0].set_title(f'Signal and noisy samples (sigma2={sigma2})')

    axs[1, 1].plot(sin, 'b-')
    axs[1, 1].plot(sin_hat, 'r--')
    axs[1, 1].set_title(f'True f0=0.0675 (blue) and estimated f0={f_hat} (red)')

    plt.show()
    
    
if __name__ == '__main__':
    main()