import numpy as np
from sklearn.datasets import load_iris
import irls
from time import time


def main():
    data = load_iris()
    X = np.c_[np.ones(data['data'].shape[0]-50), data['data'][50:]]
    y = data['target'][50:]
    y -= 1
    y = y.astype(np.int)
    it = 1000

    start = time()
    for i in range(it):
        irls.IRLS_poi(X, y, 1e-08)
    dur = time() - start
    print(dur)
    print(np.array(irls.IRLS_poi(X, y, 1e-08).round(3)))


if __name__ == '__main__':
    main()
