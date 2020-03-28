import matplotlib.pyplot as plt
import numpy as np

def line_plot(ax, data, title, x_lim, y_lim):
    ax.clear()
    ax.plot(data)
    ax.set_title(title)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

def decision_region(ax, X, Y, network):
    ax.clear()
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_min -= (x_max - x_min) * 0.01
    x_max += (x_max - x_min) * 0.01
    y_min -= (y_max - y_min) * 0.01
    y_max += (y_max - y_min) * 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 200), np.arange(y_min, y_max, (y_max - y_min) / 200))
    Z = np.array([xx.ravel(), yy.ravel()]).T
    Z = np.argmax(network.forward(Z), axis=1).reshape(xx.shape)

    Y = np.argmax(Y, axis=1).reshape(-1, 1)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.jet)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=3, cmap=plt.cm.jet)
    plt.axis('scaled')




def gt_pd(X, gt_Y, pd_Y, network):
    """ Data visualization with ground truth and predicted data comparison. There are two plots
    for them and each of them use different colors to differentiate the data with different labels.

    Args:
        data:   the input data
        gt_Y:   ground truth to the data
        pd_Y: predicted results to the data
    """
    assert X.shape[0] == gt_Y.shape[0]
    assert X.shape[0] == pd_Y.shape[0]

    gt_Y = gt_Y.flatten()

    plt.title('Ground Truth', fontsize=18)
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=gt_Y, s=3, cmap=plt.cm.jet)
    plt.axis('scaled')

    plt.subplot(1, 2, 2)
    plt.title('Prediction', fontsize=18)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 500), np.arange(y_min, y_max, (y_max - y_min) / 500))
    Z = np.array([xx.ravel(), yy.ravel()]).T
    Z = np.argmax(network.forward(Z), axis=1).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.jet)
    plt.scatter(X[:, 0], X[:, 1], c=gt_Y, s=3, cmap=plt.cm.jet)
    plt.axis('scaled')
    plt.show()
