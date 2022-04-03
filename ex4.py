import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


def main():

    n_iter = 100
    lr = 0.001
    x, y = loadData()

    # Task1
    # Initial logistic regression model
    clf1 = LogisticRegression().fit(x, y)
    
    # Logistic regression model with no penalty term or intercept
    clf2 = LogisticRegression(penalty='none', fit_intercept=False).fit(x,y)
    

    # Task2: Own SSE gradient rule linear regression
    w_SSE = np.transpose([1, -1])
    weights_SSE = []
    weights_SSE.append(w_SSE)
    accs_SSE = []
    
    for i in range(n_iter):
        for n in range(len(x)):
            w_SSE = w_SSE - lr * gradient_SSE(x[n], y[n], w_SSE)
            acc_SSE, _ = predict_(w_SSE, x, y)
        weights_SSE.append(w_SSE)
        accs_SSE.append(acc_SSE)
    weights_SSE = np.array(weights_SSE)
    accs_SSE = np.array(accs_SSE)


    # Task3: Own ML gradient rule linear regression
    w_ML = np.transpose([1, -1])
    weigths_ML = []
    weigths_ML.append(w_ML)
    accs_ML = []

    for i in range(n_iter):
        w_ML = w_ML + lr * x.T.dot(y - sigmoid(x.dot(w_ML)))
        acc_ML, _ = predict_(w_ML, x, y)
        weigths_ML.append(w_ML)
        accs_ML.append(acc_ML)
    weigths_ML = np.array(weigths_ML)
    accs_ML = np.array(accs_ML)


    # Print results
    print(f"\nCoefficients for SKlearn lr1: {clf1.coef_}")
    print(f"Accuracy for lr1: {clf1.score(x,y)}")

    print(f"\nCoefficients for SKlearn lr2: {clf2.coef_}")
    print(f"Accuracy for lr2: {clf2.score(x,y)}")

    print(f"\nCoefficients for SSE LR: {w_SSE}")
    print(f"Accuracy for own SSE LR: {acc_SSE}")

    print(f"\nCoefficients for ML LR: {w_ML}")
    print(f"Accuracy for own ML LR: {acc_ML}")


    # Plot results
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(weights_SSE[:,0], weights_SSE[:,1], 'co-')
    axs[0].plot(weigths_ML[:,0], weigths_ML[:,1], 'ro-')
    axs[0].plot(clf2.coef_[:,0], clf2.coef_[:,1], color='k', marker='x')
    axs[0].set_title("Optimization path")
    axs[0].set_xlabel("w1")
    axs[0].set_ylabel("w2")

    axs[1].plot(accs_SSE, color='c')
    axs[1].plot(accs_ML, color='r')
    axs[1].axhline(clf2.score(x,y), 0, 100, color='k', linestyle='dashed')
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Accuracy / %")

    fig.legend(["SSE", "ML", "SKlearn"])
    plt.show()


def gradient_SSE(x,y,w):
    """
    Calculate the gradient for SSE loss
    """
    return np.sum(-2*(y-sigmoid(np.sum(w*x)))*(1-sigmoid(np.sum(w*x)))*sigmoid(np.sum(w*x)))*x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_(w, x, y, probab_threshold=0.5):
    """
    Calculate predictions by using optimized weights
    """
    predicted_classes = (sigmoid(np.dot(x, w[:, np.newaxis])) >= probab_threshold).astype(int).flatten()
    accuracy = np.mean(predicted_classes == y)
    return accuracy, predicted_classes

def loadData():
    # Load and normalize data
    x = np.transpose(np.loadtxt('data/X.dat', unpack = True))
    y = np.loadtxt('data/y.dat', unpack = True)
    mean1 = np.mean(x[:,0])
    mean2 = np.mean(x[:,1])
    for i in range(len(x)):
        x[i,0] = x[i,0] - mean1
        x[i,1] = x[i,1] - mean2
    y[y==-1] = 0
    return x, y


if __name__ == '__main__':
    main()