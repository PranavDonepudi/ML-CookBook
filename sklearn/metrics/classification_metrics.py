"""Classification metrics.
With some Models for example
Date: 12-08-2025
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import _california_housing

knn = KNeighborsClassifier(n_neighbors=6)
