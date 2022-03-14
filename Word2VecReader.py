import numpy as np
import gensim
from sklearn.cluster import KMeans
import tensorflow as tf


class W2VEmbReader:
    def __init__(self, emb_path):
        emb_matrix = []
        model = gensim.models.Word2Vec.load(emb_path)

        for word in model.wv.key_to_index:
            emb_matrix.append(list(model.wv[word]))

        self.emb_matrix = np.asarray(emb_matrix)
        self.model = model

    def get_emb_matrix_given_vocab(self, vocab, emb_matrix):
        counter = 0.
        for word, index in vocab.items():
            try:
                emb_matrix[index] = list(self.model.wv[word])
                counter += 1
            except KeyError:
                pass

        print('%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100 * counter / len(vocab)))
        norm_emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=-1, keepdims=True)
        return norm_emb_matrix

    def get_model(self):
        return self.model

    def get_aspect_matrix(self, n_clusters):
        km = KMeans(n_clusters=n_clusters)
        km.fit(self.emb_matrix)
        clusters = km.cluster_centers_
        norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)
        return norm_aspect_matrix.astype(np.float32)


