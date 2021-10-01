import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def export_tsne(x, y, filename):
    model = TSNE(n_components=2, random_state=0)
    x_fitted = model.fit_transform(x)

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(111)
    plt.scatter(x_fitted[:,0], x_fitted[:,1], c=y, cmap=plt.cm.Spectral)
    fig.savefig(filename)
