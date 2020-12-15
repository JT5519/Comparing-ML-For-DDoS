import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
iris= pd.read_csv('UDP.csv')
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
X=iris[A]
iris['Class']= le.fit_transform(iris['Class'])
y=iris['Class'] 
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import MultinomialNB
from mlxtend.classifier import StackingClassifier
import numpy as np
import time as ti
import warnings
warnings.simplefilter('ignore')
clf1 = OneVsRestClassifier(LogisticRegression())
#clf1 = OneVsRestClassifier(GaussianNB())
#clf1 = OneVsRestClassifier(RandomForestClassifier())
clf2 = BaggingClassifier()  
#clf2 = BaggingClassifier(max_samples=200) 
#clf2 = BaggingClassifier(n_estimators=5,max_samples=200)
clf3 = MultinomialNB()
#clf3 = MultinomialNB(alpha=2)
#clf3 = MultinomialNB(alpha=2,fit_prior=False)
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],meta_classifier=lr)
for clf, label in zip([clf1, clf2, clf3, sclf],['OneVsRestClassifier', 'Bagging Classifier', 'MultinomialNB','Ensemble']):
    start_time = ti.time()
    scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    end_time = ti.time()
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    print("Time taken: %0.2f seconds"%(end_time-start_time))
    print()




