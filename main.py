from sklearn import neighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import os

from helpers import apply_cross_validation_and_evaluate


def remove_stopping_words(data):
   # filtered_data = [word for word in data if word not in stopwords.words('english')]
    return data


def read_file_in_dir(directory):
    docs = []
    with os.scandir(directory) as entries:
        for entry in entries:
            file = open(directory + "/"  + entry.name, 'r')
            data = file.read().replace("\n", "")
            docs.append(remove_stopping_words(data))
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

df_idf = pd.DataFrame(tfs[0].T.todense(), index=Tfidf_Vectorizer.vocabulary_,columns=["idf_weights"])
df_idf.sort_values(by=['idf_weights'] , ascending=False)

print(tfs.shape)
x_data = tfs.reshape((tfs.shape[0], tfs.shape[1], tfs.shape[1] ,1))
print(x_data.shape)

#x_train = tfs.reshape(tfs.shape[0], tfs.shape[1],1)
#Input(batch_shape=(None, tf_len, 1))
'''
print(tfs.shape)

x_data = np.array(tfs)

print(x_data[0][0])
'''
