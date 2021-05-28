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
tfs_training = Tfidf_Vectorizer.fit_transform(x_train).astype('float64')


# Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel


# Train the model using the training sets
clf.fit(tfs_training, y_train)




# Predict the response for test dataset
tfs_testing = Tfidf_Vectorizer.transform(x_test).astype('float64').todense()
y_pred = clf.predict(tfs_testing)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#case = Tfidf_Vectorizer.transform(["Iam very Sad"]).astype('float64')
#print(clf.predict(case))

vec =  clf.support_vectors_


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

clf = TruncatedSVD(1000)
Xpca = clf.fit_transform(tfs_testing)

pca = PCA()
pca.fit_transform(Xpca)
pca_variance = pca.explained_variance_

pca2 = PCA(n_components=10)
pca2.fit(Xpca)
x_3d = pca2.transform(Xpca)

# Creating figure
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection="3d")

# Creating plot
ax.scatter3D(x_3d[:,0], x_3d[:,5], x_3d[:,9], c=y_test)
plt.title("simple 3D scatter plot")

# show plot
plt.show()