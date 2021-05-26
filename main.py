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


list_of_zeros = [0] * 1000
list_of_ones = [1] * 1000

y_data = list_of_ones + list_of_zeros

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
