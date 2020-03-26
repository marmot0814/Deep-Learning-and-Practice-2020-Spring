import matplotlib.pyplot as plt

def gt_pd(data, gt_y, pd_y, c = ['r', 'b']):
    """ Data visualization with ground truth and predicted data comparison. There are two plots
    for them and each of them use different colors to differentiate the data with different labels.

    Args:
        data:   the input data
        gt_y:   ground truth to the data
        pd_y: predicted results to the data
    """
    assert data.shape[0] == gt_y.shape[0]
    assert data.shape[0] == pd_y.shape[0]

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize=18)

    for x, y in zip(data, gt_y):
        plt.plot(x[0], x[1], c[y[0]] + 'o')
    plt.axis('scaled')

    plt.subplot(1, 2, 2)
    plt.title('Prediction', fontsize=18)

    for x, y in zip(data, pd_y):
        plt.plot(x[0], x[1], c[y[0]] + 'o')
    plt.axis('scaled')

    plt.show()
