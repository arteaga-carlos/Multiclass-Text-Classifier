#!/usr/bin/env python
# coding: utf-8


import pandas as pd

df = pd.read_csv('_dataSet30k.csv')
df.head()



from io import StringIO

# get only the columns we need
col = ['issueKey', 'combined']
df = df[col]

# remove null rows
df = df[pd.notnull(df['combined'])]

df.columns = ['issueKey', 'combined']

# create new column representing issueKey as an integer
df['category_id'] = df['issueKey'].factorize()[0]

category_id_df = df[['issueKey', 'category_id']].drop_duplicates().sort_values('category_id')

category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'issueKey']].values)

df.head(7)



# see data distribution
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6))

df.groupby('issueKey').combined.count().plot.bar(ylim=0)

plt.show()



# start extracting features from text rows
from sklearn.feature_extraction.text import TfidfVectorizer

# Term Frequency, Inverse Document Frequency
# vector for each row
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

# fit and transform
features = tfidf.fit_transform(df.combined).toarray()

#labels
labels = df.category_id

features.shape



# train best performing model from above
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['combined'], df['issueKey'], random_state = 0)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

model = MultinomialNB()

clf = model.fit(X_train_tfidf, y_train)



# make DataFrame containing cross validation accuracies (cv_df)
from sklearn.model_selection import cross_val_score

# CV = cross validation. In this case it contains number of folds
CV = 5
cv_df = pd.DataFrame(index=range(CV))
entries = []

accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)

model_name = model.__class__.__name__

for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])



print(clf.predict(count_vect.transform(["test paragraph"])))


cv_df.accuracy.mean()


# save model
import pickle

# Save to file in the current working directory
pickle_filename = "pickle_model.pkl"

with open(pickle_filename, 'wb') as file:  
    pickle.dump(clf, file)
    
# Load from file
with open(pickle_filename, 'rb') as file:  
    pickle_clf = pickle.load(file)
    
# Use loaded model
print(pickle_clf.predict(count_vect.transform(["pickle test paragraph"])))

