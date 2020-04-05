import matplotlib.pyplot as plt
import numpy as np

def line_plot(ax, data, title, y_lim):

    ax.clear()
    ax.plot(range(len(data)), data, alpha=0.3, c='b')

    y = data[0]
    smooth = []
    for x in data:
        y = 0.3 * y + 0.7 * x
        smooth.append(y)
    ax.plot(range(len(smooth)), smooth, c='b', alpha=0.7)

    ax.plot(range(len(smooth))[::10], smooth[::10], color='orange', marker='o', linestyle='dashed')
    ax.set_title(title)
    ax.set_xlim([0, max(50, len(data))])
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
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, cmap=plt.cm.jet)
    plt.axis('scaled')




def gt_vs_pd(X, gt_Y, pd_Y, network):
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

    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize=18)
    plt.scatter(X[:, 0], X[:, 1], c=gt_Y, s=20, cmap=plt.cm.jet)
    plt.axis('scaled')

    plt.subplot(1, 2, 2)
    plt.title('Prediction', fontsize=18)
    plt.scatter(X[:, 0], X[:, 1], c=pd_Y, s=20, cmap=plt.cm.jet)
    plt.axis('scaled')
    plt.show()
