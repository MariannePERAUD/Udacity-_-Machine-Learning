#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################


from sklearn import svm
from sklearn.metrics import accuracy_score



# the classifier
clf = svm.SVC(kernel="rbf",C=10000)

t0=time()
# train
"""
added to slice training dataset and reduce by 100

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]
"""


clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

# predict
t1=time()
pred = clf.predict(features_test)
print "prediction time", round(time()-t1, 3), "s"

accuracy = accuracy_score(pred, labels_test)
print '\naccuracy = {0}'.format(accuracy)

f10=pred[10]
f26=pred[26]
f50=pred[50]

print(f10,f26,f50)

print(type(pred))
print((pred==0).sum())
print((pred==1).sum())

"""
WITH SVM and WHOlE FILE :
In [2]: runfile('/Users/mperaud/notebooks/ud120-projects-master/svm/svm_author_id.py', wdir='/Users/mperaud/notebooks/ud120-projects-master/svm')
Reloaded modules: email_preprocess
no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time: 168.821 s
prediction time 18.267 s

accuracy = 0.984072810011

WITH NAIVE BAYES
In [3]: runfile('/Users/mperaud/notebooks/ud120-projects-master/naive_bayes/nb_author_id.py', wdir='/Users/mperaud/notebooks/ud120-projects-master/naive_bayes')
Reloaded modules: email_preprocess
no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time: 2.222 s
prediction time 0.225 s

accuracy = 0.973265073948

WITH SVM(linear) and SMALLER DATASET
runfile('/Users/mperaud/notebooks/ud120-projects-master/svm/svm_author_id.py', wdir='/Users/mperaud/notebooks/ud120-projects-master/svm')
Reloaded modules: email_preprocess
no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time: 0.125 s
prediction time 1.199 s

accuracy = 0.884527872582

WITH SVM(rbf) and SMALLER DATASET C=1
runfile('/Users/mperaud/notebooks/ud120-projects-master/svm/svm_author_id.py', wdir='/Users/mperaud/notebooks/ud120-projects-master/svm')
Reloaded modules: email_preprocess
no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time: 0.109 s
prediction time 1.256 s

accuracy = 0.616040955631

WITH SVM(rbf) and SMALLER DATASET C=10000
runfile('/Users/mperaud/notebooks/ud120-projects-master/svm/svm_author_id.py', wdir='/Users/mperaud/notebooks/ud120-projects-master/svm')
Reloaded modules: email_preprocess
no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time: 0.113 s
prediction time 1.019 s

accuracy = 0.892491467577

WHOLE DATASET WITH SVM(rbf) C=10000
runfile('/Users/mperaud/notebooks/ud120-projects-master/svm/svm_author_id.py', wdir='/Users/mperaud/notebooks/ud120-projects-master/svm')
Reloaded modules: email_preprocess
no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time: 125.825 s
prediction time 13.304 s

accuracy = 0.990898748578
"""
