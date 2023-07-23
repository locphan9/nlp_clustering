import pandas as pd
import re
import string

import nltk
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns

df =  pd.read_csv('data.csv')
df = df.dropna(subset=['title'])


# Load spacy
nlp = spacy.load('en_core_web_sm')

def clean_string(text):

    final_string = ""

    # Make lower
    text = text.lower()

    # Remove line breaks
    text = re.sub('\n', '', text)

    # Remove puncuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Lemmatize
    text = nlp(text)
    text = [y.lemma_ for y in text]
    
    # Convert to list
    # text = text.split()
    
    # Remove stop words
    useless_words = nltk.corpus.stopwords.words("english")
    

    text_filtered = [word for word in text if not word in useless_words]

    # Remove numbers
    text_filtered = [re.sub('\w*\d\w*', '', w) for w in text_filtered]


    final_string = ' '.join(text_filtered)

    return final_string

# Next apply the clean_string function to the text
df['title_clean'] = df['title'].apply(lambda x: clean_string(x))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['title_clean'])
Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(init="k-means++", n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)

plt.figure(figsize=(7,5))
ax = sns.lineplot(x=K, y=Sum_of_squared_distances)
#732F2F
ax.lines[0].set_linestyle("--")
ax.lines[0].set_color("#F25D27")

# Add a vertical line to show the optimum number of clusters
plt.axvline(2, color='#1A2226', linestyle=':')

plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal k')

plt.tight_layout()

plt.savefig('./textclustering_elbow.png', dpi=300)

# Set the number of clusters
k = 2
# Vectorize the text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['title_clean'])
# Fit our Model
model = KMeans(init="k-means++", n_clusters=k, max_iter=25, n_init=1)
model.fit(X)

# Get the cluster labels
clust_labels = model.predict(X)
cent = model.cluster_centers_

kmeans_labels = pd.DataFrame(clust_labels)
df.insert((df.shape[1]),'clusters',kmeans_labels)

df.sample(5, random_state=3)

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

results_dict = {}

for i in range(k):
    terms_list = []
    
    for ind in order_centroids[i, :15]:  
        terms_list.append(terms[ind])
    
    results_dict[f'Cluster {i}'] = terms_list
    
df_clusters = pd.DataFrame.from_dict(results_dict)
print(df_clusters)