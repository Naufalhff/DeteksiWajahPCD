import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from itertools import combinations
import random

# Baca embeddings dari CSV
df = pd.read_csv('face_embeddings.csv')

# Visualisasi t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_result = tsne.fit_transform(df.iloc[:, 3:])  # Kolom fitur mulai dari index ke-3

df['tsne-2d-one'] = tsne_result[:, 0]
df['tsne-2d-two'] = tsne_result[:, 1]

plt.figure(figsize=(10, 6))
for person in df['person'].unique():
    subset = df[df['person'] == person]
    plt.scatter(subset['tsne-2d-one'], subset['tsne-2d-two'], label=person)

plt.legend()
plt.title('t-SNE Visualization of Face Embeddings')
plt.show()

# === Generate Positif dan Negatif Pairs ===
positive_pairs = []
negative_pairs = []

# Buat pasangan positif (orang yang sama)
for person in df['person'].unique():
    samples = df[df['person'] == person]
    if len(samples) >= 2:
        comb = list(combinations(samples.index, 2))
        for i, j in comb:
            positive_pairs.append((df.loc[i], df.loc[j]))

# Buat pasangan negatif (orang berbeda)
while len(negative_pairs) < len(positive_pairs):
    i, j = random.sample(range(len(df)), 2)
    if df.iloc[i]['person'] != df.iloc[j]['person']:
        negative_pairs.append((df.iloc[i], df.iloc[j]))

# === Hitung cosine similarity dan label ===
def get_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

similarities = []
labels = []

# Positif: label 1
for pair in positive_pairs:
    sim = get_similarity(pair[0].iloc[3:], pair[1].iloc[3:])
    similarities.append(sim)
    labels.append(1)

# Negatif: label 0
for pair in negative_pairs:
    sim = get_similarity(pair[0].iloc[3:], pair[1].iloc[3:])
    similarities.append(sim)
    labels.append(0)

# === ROC Curve ===
fpr, tpr, thresholds = roc_curve(labels, similarities)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# === Evaluasi Tambahan ===
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
y_pred = [1 if s >= optimal_threshold else 0 for s in similarities]

print("\n=== Evaluation Metrics ===")
print("Optimal Threshold:", optimal_threshold)
print(confusion_matrix(labels, y_pred))
print(classification_report(labels, y_pred, target_names=["Different Person", "Same Person"]))
