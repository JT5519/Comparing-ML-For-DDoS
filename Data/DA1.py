# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import time as ti
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import operator
import pandas as pd
#iris = datasets.load_iris()
#X, y = iris.data[:, 1:3], iris.target
iris = pd.read_csv('UDP.csv')
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
iris['UTC Time'] = le.fit_transform(iris['UTC Time'])
iris['Absolute Time'] = le.fit_transform(iris['Absolute Time'])
iris['Source'] = le.fit_transform(iris['Source'])
iris['Destination'] = le.fit_transform(iris['Destination'])
iris['Protocol'] = le.fit_transform(iris['Protocol'])
iris['SourcePort'] = le.fit_transform(iris['SourcePort'])
iris['DestPort'] = le.fit_transform(iris['DestPort'])
iris['Hwdestaddr'] = le.fit_transform(iris['Hwdestaddr'])
iris['Hwsrcaddr'] = le.fit_transform(iris['Hwsrcaddr'])
iris['Unresolved Destport'] = le.fit_transform(iris['Unresolved Destport'])
iris['Unresolved Srcport'] = le.fit_transform(iris['Unresolved Srcport'])
#iris['NetSrcAddr'] = le.fit_transform(iris['NetSrcAddr'])
#iris['NetDestAddr'] = le.fit_transform(iris['NetDestAddr'])
A = ['Delta Time','Length','Cumulative Bytes','Time','UTC Time','Absolute Time','Source','Destination','Protocol','SourcePort','DestPort','Hwsrcaddr','Hwdestaddr','Unresolved Destport','Unresolved Srcport']
X = iris[A]
y = iris['Class']
np.random.seed(123)
clf1 = OneVsRestClassifier(LogisticRegression())
clf2 = BaggingClassifier()  
clf3 = MultinomialNB()
for clf, label in zip([clf1, clf2, clf3], ['OneVsRestClassifier', 'Bagging Classifier', 'MultinomialNB']):

    start_time = ti.time()
    scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    end_time = ti.time()
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    print("Time taken: %0.2f seconds"%(end_time-start_time))
    print()
