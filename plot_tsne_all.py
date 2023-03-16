# code to plot the t-SNE of acoustic embeddings from w2v2 with their labels
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

# load data and labels
features_w2v2 = pd.read_pickle("cache/w2v2.pkl")
df = pd.read_csv("EmoFilm_labels_16k_split.csv")

# map emotion label from integer to string
emotion_map = {
    0: "fear", #"fear",
    1: "contempt", #"disgust",
    2: "happines", #"happiness",
    3: "anger", #"anger",
    4: "sadness", #"sadness",
}

df['emotions'] = df['emo'].map(emotion_map) # add a new column with emotion string labels

tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features_w2v2)

# set style to solarized
plt.style.use('Solarize_Light2')
# plot the t-SNE with the emotion labels
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=df['emotions'], legend='full')
plt.show()