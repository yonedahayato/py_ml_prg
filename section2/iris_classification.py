import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perceptron import Perceptron

class Iris(object):
    def __init__(self):
        data_df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
        print(data_df.tail())

        y = data_df.iloc[0:100, 4].values
        y = np.where(y == "Iris-setosa", -1, 1)

        X = data_df.iloc[0:100, [0, 2]].values

        plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")
        plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="setosa")

        plt.savefig('iris.png')
        plt.figure()

        self.y = y
        self.x = X

    def classification(self):
        ppt = Perceptron(eta=0.1, n_iter=10)

        ppt.fit(self.x, self.y)

        plt.plot(range(1, len(ppt.errors_)+1), ppt.errors_, marker="o")
        plt.savefig("miss_classification.png")


def main():
    iris = Iris()
    iris.classification()

if __name__ == "__main__":
    main()
