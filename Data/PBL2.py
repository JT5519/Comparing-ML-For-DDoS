# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator
class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier for scikit-learn estimators.

    Parameters
    ----------

    clf : `iterable`
      A list of scikit-learn classifier objects.
    weights : `list` (default: `None`)
      If `None`, the majority rule voting will be applied to the predicted class labels.
        If a list of weights (`float` or `int`) is provided, the averaged raw probabilities (via `predict_proba`)
        will be used to determine the most confident class label.

    """
    def __init__(self, clfs, weights=None):
        self.clfs = clfs
        self.weights = weights

    def fit(self, X, y):
        """
        Fit the scikit-learn estimators.

        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels

        """
        for clf in self.clfs:
            clf.fit(X, y)

    def predict(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule

        """

        self.classes_ = np.asarray([clf.predict(X) for clf in self.clfs])
        if self.weights:
            avg = self.predict_proba(X)

            maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)

        else:
            maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])

        return maj

    def predict_proba(self, X):

        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        avg : list or numpy array, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.

        """
        self.probas_ = [clf.predict_proba(X) for clf in self.clfs]
        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg


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
eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3],weights=[1,1,1])

for clf, label in zip([clf1, clf2, clf3,eclf], ['OneVsRestClassifier', 'Bagging Classifier', 'MultinomialNB','Ensemble']):
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

