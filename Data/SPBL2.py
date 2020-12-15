# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator
from sklearn import datasets
#iris = datasets.load_iris()
import pandas as pd
iris = pd.read_csv('UDP.csv')
#X, y = iris.data[:, 1:3], iris.target
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
y= le.fit_transform(y)
from sklearn import model_selection
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import numpy as np
from sklearn import metrics
import time as ti
np.random.seed(123)
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.80,random_state=42)
#clf1 = OneVsRestClassifier(LogisticRegression())
#clf1 = OneVsRestClassifier(GaussianNB())
clf1 = OneVsRestClassifier(RandomForestClassifier())
#clf2 = BaggingClassifier()  
#clf2 = BaggingClassifier(max_samples=200) 
clf2 = BaggingClassifier(n_estimators=5,max_samples=200)
#clf3 = MultinomialNB()
#clf3 = MultinomialNB(alpha=2)
clf3 = MultinomialNB(alpha=2,fit_prior=False)
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],meta_classifier=lr)
for clf, label in zip([clf1, clf2, clf3,sclf],['OneVsRestClassifier', 'Bagging Classifier', 'MultinomialNB','Ensemble']):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    TP=0
    TN=0
    FP=0
    FN=0
    for a,b in zip(y_test,y_pred):
        if(a==0 and b==0):
            TP = TP+1
        elif(a==0 and b==1):
            FN = FN+1
        elif(a==1 and b==0):    
            FP = FP+1
        else:
            TN = TN+1
    scores1 = metrics.precision_recall_fscore_support(y_test,y_pred,average=None)
    scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    print("Precision: %0.2f" %scores1[0].mean())
    print("Recall: %0.2f" %scores1[1].mean())
    print("F-measure: %0.2f" %scores1[2].mean())
    print("True positives: %d"%TP)
    print("True Negatives: %d"%TN)
    print("False positives: %d"%FP)
    print()




