#! /usr/bin/env python3

"""
Iris Flower Data Set Machine Learning Problem

Wikipedia Page
https://en.wikipedia.org/wiki/Iris_flower_data_set

K-Nearest Neighbors Algorithm
https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

Iris plants dataset
--------------------

**Data Set Characteristics:**

    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica

    :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988

The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
from Fisher's paper. Note that it's the same as in R, but not as in the UCI
Machine Learning Repository, which has two wrong data points.

This is perhaps the best known database to be found in the
pattern recognition literature.  Fisher's paper is a classic in the field and
is referenced frequently to this day.  (See Duda & Hart, for example.)  The
data set contains 3 classes of 50 instances each, where each class refers to a
type of iris plant.  One class is linearly separable from the other 2; the
latter are NOT linearly separable from each other.

.. topic:: References

   - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
     Mathematical Statistics" (John Wiley, NY, 1950).
   - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
     Structure and Classification Rule for Recognition in Partially Exposed
     Environments".  IEEE Transactions on Pattern Analysis and Machine
     Intelligence, Vol. PAMI-2, No. 1, 67-71.
   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
     on Information Theory, May 1972, 431-433.
   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
     conceptual clustering system finds 3 classes in the data.
   - Many, many more ...

"""

from dataclasses import dataclass
from typing import Tuple, Dict
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import Bunch

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SepalData:
    """ Class to store relevant sepal data """
    sepal_length: np.ndarray
    sepal_width: np.ndarray
    sepal_length_label: str
    sepal_width_label: str


@dataclass
class PetalData:
    """ Class to store relevant petal data """
    petal_length: np.ndarray
    petal_width: np.ndarray
    petal_length_label: str
    petal_width_label: str


class KNN:
    """ Manage KNN Model """
    def __init__(self, n_neighbors: int = 1) -> None:
        self.iris_types: Dict[int, str] = {
            0: "Iris Setosa",
            1: "Iris Versicolor",
            2: "Iris Virginica"
        }
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.x_train: np.ndarray = None
        self.x_test: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None

    def split_testing_data(self, data: np.ndarray, target: np.ndarray,
                           random_state: int = 0) -> None:
        """ Split data for testing and training the model """
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(data, target, random_state=random_state)

    def fit_data(self) -> None:
        """ Fit data to our KNN model """
        self.knn.fit(self.x_train, self.y_train)

    def score_model(self) -> float:
        """ Return the current model score for accuracy """
        return self.knn.score(self.x_test, self.y_test)

    def predict_type(self, new_value: np.array) -> Tuple[str, int]:
        """ Predict the type of Iris using KNN model """
        prediction: np.ndarray = self.knn.predict(new_value)
        return prediction[0], self.iris_types.get(prediction[0])


class ScatterPlot:
    """ Manage Scatter Plot Graphs """
    def __init__(self, petal_data: PetalData, sepal_data: SepalData,
                 target: np.ndarray) -> None:
        self.fig: plt.Figure = plt.figure()
        self.add_scatter(x_axis=(sepal_data.sepal_length_label,
                                 sepal_data.sepal_length),
                         y_axis=(sepal_data.sepal_width_label,
                                 sepal_data.sepal_width),
                         target=target,
                         position=121)
        self.add_scatter(x_axis=(petal_data.petal_length_label,
                                 petal_data.petal_length),
                         y_axis=(petal_data.petal_width_label,
                                 petal_data.petal_width),
                         target=target,
                         position=122)
        self.display_scatter()

    def add_scatter(self, x_axis: Tuple[str, np.ndarray],
                    y_axis: Tuple[str, np.ndarray],
                    target: np.ndarray, position: int) -> None:
        """ Add scatter plot to overall graph """
        sub: plt.Subplot = self.fig.add_subplot(position)
        sub.scatter(x=x_axis[1], y=y_axis[1], c=target)
        sub.set_xlabel(x_axis[0])
        sub.set_ylabel(y_axis[0])

    @staticmethod
    def display_scatter() -> None:
        """ Display scatter plots """
        plt.show()


def main() -> None:
    """ Main application method """
    iris_data: Bunch = load_iris()
    knn: KNN = KNN()
    knn.split_testing_data(data=iris_data["data"], target=iris_data["target"])
    knn.fit_data()
    print(knn.score_model())
    print(knn.predict_type(new_value=np.array([[5.0, 2.9, 1.0, 0.2]])))

    transposed: np.ndarray = iris_data["data"].T
    sepal_data: SepalData = SepalData(sepal_length=transposed[0],
                                      sepal_width=transposed[1],
                                      sepal_length_label=iris_data[
                                          "feature_names"][0],
                                      sepal_width_label=iris_data[
                                          "feature_names"][1])
    petal_data: PetalData = PetalData(petal_length=transposed[2],
                                      petal_width=transposed[3],
                                      petal_length_label=iris_data[
                                          "feature_names"][2],
                                      petal_width_label=iris_data[
                                          "feature_names"][3])
    ScatterPlot(petal_data=petal_data, sepal_data=sepal_data,
                target=iris_data["target"])


if __name__ == '__main__':
    main()
