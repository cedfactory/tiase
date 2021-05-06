import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def ExportTSNE(X, y, filename):
    #model = TSNE(perplexity = 50, learning_rate = 400, n_iter = 2000, n_iter_without_progress = 100)
    model = TSNE(n_components=2, random_state=0)
    npX = np.array(X)
    Xfitted = model.fit_transform(X)

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(111)
    plt.scatter(Xfitted[:,0], Xfitted[:,1], c=y, cmap=plt.cm.Spectral)
    fig.savefig(filename)
