# plot t-SNE of English data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

# load data and labels
features_w2v2 = pd.read_pickle("cache/w2v2.pkl")
df = pd.read_csv("EmoFilm_labels_16k_split.csv")

# map emotion label from integer to string
emotion_map = {
    0: "fear",
    1: "contempt",
    2: "happines",
    3: "anger",
    4: "sadness",
}

# use mono language only data: es, it, en
df_en = df[df['language'] == 'es']
df_en['emotions'] = df_en['emo'].map(emotion_map) # add a new column with emotion string labels

# Filter only English data
en_files = df_en['file'].tolist()
en_features_w2v2 = features_w2v2.loc[en_files]

tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(en_features_w2v2)

# set style to solarized
plt.style.use('Solarize_Light2')
# plot the t-SNE with the emotion labels for English language
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=df_en['emotions'], legend='full')
plt.show()