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
    0: "fe", #"fear",
    1: "co", #"disgust",
    2: "ha", #"happiness",
    3: "an", #"anger",
    4: "sa", #"sadness",
}

tsne = TSNE(n_components=2, random_state=42) #, init='pca', learning_rate='auto')
tsne_results = tsne.fit_transform(features_w2v2)

# plot the t-SNE with the emotion labels
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=df['emo'])
# show legend with emotion labels
# plt.legend(handles=[plt.scatter([],[],c=i, label=emotion_map[i]) for i in emotion_map.keys()], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()