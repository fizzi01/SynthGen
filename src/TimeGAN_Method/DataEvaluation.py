from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class DataEvaluation:

    def __init__(self, real_data: list, synth_data: list, seq_len: int = 24, sample_size: int = 250):
        self.real_data = real_data
        self.synth_data = synth_data
        self.seq_len = seq_len
        self.sample_size = sample_size
        self.real_sample = self._real_sample()
        self.synthetic_sample = self._syntethic_sample()

    def _real_sample(self):
        idx = np.random.permutation(len(self.real_data))[:self.sample_size]

        # Convert list to array, but taking only {sample_size} random samples chosen in idx
        # data: (list(rows(ndarray(24, columns)))) -> real_sample: ndarray(250, 24, columns)
        return np.asarray(self.real_data)[idx]

    def _syntethic_sample(self):
        idx = np.random.permutation(len(self.real_data))[:self.sample_size]

        # Convert list to array, but taking only {sample_size} random samples chosen in idx
        # data: (list(rows(ndarray(24, columns)))) -> real_sample: ndarray(250, 24, columns)
        return np.asarray(self.synth_data)[idx]

    def _reduce_sample(self):
        # For the purpose of comparison we need the data to be 2-Dimensional. array (250*columns,seq_lenght)
        synth_data_reduced = self.real_sample.reshape(-1, self.seq_len)
        real_data_reduced = np.asarray(self.synthetic_sample).reshape(-1, self.seq_len)

        self.synthetic_sample = synth_data_reduced
        self.real_sample = real_data_reduced

    def evaluate_data(self):
        self._reduce_sample()

        dfs = self._evaluation(n_components=2)
        self._plot_evaluation(dfs[0], dfs[1], dfs[2])

    def _evaluation(self, n_components: int):
        pca = PCA(n_components=n_components)
        tsne = TSNE(n_components=n_components, n_iter=300)
        pca.fit(self.real_sample)

        pca_real = pd.DataFrame(pca.transform(self.real_sample))
        pca_synth = pd.DataFrame(pca.transform(self.synthetic_sample))

        # data_reduced: {ndarray: (250*columns*2, 24)}
        data_reduced = np.concatenate((self.real_sample, self.synthetic_sample), axis=0)

        # tsne_results: {DataFrame: (250*columns*2, 2)}
        tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))

        return pca_real, pca_synth, tsne_results

    def _plot_evaluation(self, pca_real: pd.DataFrame, pca_synth: pd.DataFrame, tsne_results: pd.DataFrame):
        fig = plt.figure(constrained_layout=True, figsize=(40, 20))
        spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

        ax = fig.add_subplot(spec[0, 0])
        ax.set_title('PCA results',
                     fontsize=20,
                     color='red',
                     pad=10)

        # PCA scatter plot
        plt.scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:, 1].values,
                    c='black', alpha=0.2, label='Original')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        plt.scatter(pca_synth.iloc[:, 0], pca_synth.iloc[:, 1],
                    c='red', alpha=0.2, label='Synthetic')
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        ax.legend()

        ax2 = fig.add_subplot(spec[0, 1])
        ax2.set_title('TSNE results',
                      fontsize=20,
                      color='red',
                      pad=10)

        lenght = self.real_sample.shape[0]

        # t-SNE scatter plot
        plt.scatter(tsne_results.iloc[:lenght, 0].values, tsne_results.iloc[:lenght, 1].values,
                    c='black', alpha=0.2, label='Original')
        plt.scatter(tsne_results.iloc[lenght:, 0], tsne_results.iloc[lenght:, 1],
                    c='red', alpha=0.2, label='Synthetic')

        ax2.legend()

        fig.suptitle('Validating synthetic vs real data diversity and distributions',
                     fontsize=16,
                     color='grey')
        plt.show()