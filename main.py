from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
from sklearn import svm, metrics


def read_file_in_dir(directory):
    docs = []
    with os.scandir(directory) as entries:
        for entry in entries:
            file = open(directory + "/"  + entry.name, 'r')
            data = file.read().replace("\n", "")
            docs.append(data)
    return docs


listofzeros = [0] * 1000
listofOnes = [1] * 1000

y_data = listofOnes + listofzeros

neg_docs = read_file_in_dir("neg/")
pos_docs = read_file_in_dir("pos/")
all_docs = pos_docs + neg_docs

x_train, x_test, y_train, y_test = train_test_split(all_docs, y_data)

Tfidf_Vectorizer = TfidfVectorizer(use_idf=True, stop_words='english')
tfs = Tfidf_Vectorizer.fit_transform(x_train).astype('float64')


# Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel


# Train the model using the training sets
clf.fit(tfs, y_train)


# Predict the response for test dataset
tfs = Tfidf_Vectorizer.transform(x_test).astype('float64')
y_pred = clf.predict(tfs)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#tfs = Tfidf_Vectorizer.transform(["Iam very Sad"]).astype('float64')
#print(clf.predict(tfs))


import pylab as pl

print(tfs)


for i in range(0, tfs.shape[0]):
    if y_test[i] == 1:
       # print(tfs[i,2])
        c1 = pl.scatter(tfs[i, 0], tfs[i, 1], c='r', marker='+')
    elif y_test[i] == 0:
        c2 = pl.scatter(tfs[i, 0], tfs[i, 1], c='g', marker='o')

pl.title('Iris training dataset with 3 classes and    known outcomes')
pl.show()

