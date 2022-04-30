import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

k = 3


class KNN:

    def __init__(self, k):
        self.mnist = datasets.fetch_openml('mnist_784', data_home='mnist_dataset/')
        self.data, self.target = self.mnist.data, self.mnist.target
        self.classifier = KNeighborsClassifier(n_neighbors=k)

    def skl_knn(self):
        train_x, test_x, train_y, test_y = train_test_split(self.data, self.target, test_size=0.25, random_state=42)
        self.classifier.fit(train_x, train_y)
        y_pred = self.classifier.predict(test_x)
        pickle.dump(self.classifier, open('knn.sav', 'wb'))
        print(classification_report(test_y, y_pred))
        print("KNN Classifier model saved as knn.sav!")
