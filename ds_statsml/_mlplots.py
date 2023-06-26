import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score, f1_score
from sklearn.model_selection import learning_curve, KFold, ParameterGrid
from sklearn.preprocessing import scale
from sklearn.cluster import *
from sklearn.decomposition import PCA
from IPython.display import clear_output

sns.set_style('darkgrid')
sns.set(rc={'patch.edgecolor': 'w', 'patch.force_edgecolor': True, 'patch.linewidth': 1})


def plot_feature_importances(model, featurecols, n_show=25, idx_start=0, min_importance=0, max_importance=np.inf, sortby='values', ascending=False,
                             figsize=(10, 8), print_features=None):

    features = pd.Series(index=featurecols, data=model.feature_importances_)

    if sortby == 'values':
        features_show = features[(min_importance <= features.values) &
                                 (features.values <= max_importance)].sort_values(ascending=ascending).iloc[idx_start:idx_start+n_show]
        features.sort_values(ascending=ascending, inplace=True)

    else:
        features_show = features[(min_importance <= features.values) &
                                 (features.values <= max_importance)].sort_index(ascending=ascending).iloc[idx_start:idx_start+n_show]
        features.sort_index(ascending=ascending, inplace=True)

    g = sns.barplot(x=features_show.values, y=features_show.index)
    g.figure.set_size_inches(figsize[0], figsize[1])

    if print_features is not None:
        for feature in features.index:
            if feature in print_features:
                try:
                    print(f'Feature: {feature}\t\tImportance: {features[feature]}')
                except KeyError:
                    print(f' (!) "{feature}" is not in the model feature columns.')
                    continue

    return g


def f1thresh(model, X_te, y_te):

    X_val, y_val = X_te, y_te  # explicitly calling this validation since we're using it for selection

    thresh_ps = np.linspace(.10, .50, 1000)
    model_val_probs = model.predict_proba(X_val)[:, 1]  # positive class probs, same basic logistic model we fit in section 2

    f1_scores = []
    for p in thresh_ps:
        model_val_labels = model_val_probs >= p
        f1_scores.append(f1_score(model_val_labels, y_val))

    plt.plot(thresh_ps, f1_scores)
    plt.title('F1 Score vs. Positive Class Decision Probability Threshold')
    plt.xlabel('P threshold')
    plt.ylabel('F1 score')

    best_f1_score = np.max(f1_scores)
    best_thresh_p = thresh_ps[np.argmax(f1_scores)]

    print('Logistic Regression Model best F1 score %.3f at prob decision threshold >= %.3f'
          % (best_f1_score, best_thresh_p))


def tricorr(data):
    tri_mask = np.triu(np.array(np.ones(data.corr().shape)))

    g = sns.heatmap(data.corr(), cmap='viridis', mask=tri_mask)
    g.figure.set_size_inches(10, 8)


def learningcurve(model, X, y, cv=KFold, train_sizes=np.linspace(0.1, 1.0, 10)):

    m, scores_train, scores_test = learning_curve(model, X, y, cv=cv, train_sizes=train_sizes)

    scores_train_ = np.mean(scores_train, axis=1)
    scores_test_ = np.mean(scores_test, axis=1)

    fig = plt.figure()
    plt.plot(scores_train_, label='Log. Train')
    plt.plot(scores_test_, label='Log. Test')
    plt.legend()

    return fig


def silhouette_plots(X, clusterer=MiniBatchKMeans, param_dict=None, fitted_clusterers=None, clusterers_labels=None,
                     report_on=('n_clusters')):
    # Generating the sample data from make_blobs
    # This particular setting has one distinct cluster and 3 clusters placed close
    # together.

    X = np.array(X)

    if fitted_clusterers is None:
        fitted_clusterers = []

        for params in ParameterGrid(param_dict):
            fitted_clusterer_ = clusterer(**params)
            fitted_clusterer_ = fitted_clusterer_.fit(X)
            fitted_clusterers.append(fitted_clusterer_)

    for i, fitted_clusterer in enumerate(fitted_clusterers):

        params = {k:fitted_clusterer.get_params()[k] for k in fitted_clusterer.get_params().keys() if k in report_on}

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # Initialize the clusterer
        if clusterers_labels is None:
            cluster_labels = fitted_clusterer.predict(X)

        else:
            cluster_labels = clusterers_labels[i]

        n_clusters = len(set(cluster_labels))

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        if 1 < n_clusters < X.shape[0]:
            silhouette_avg = silhouette_score(X, cluster_labels)
            print(f"For {params}, the average silhouette_score is :", silhouette_avg)
        else:
            print(f"For {params}, silhouette_score cannot be calculated. ", end="")
            print(f"n_clusters, {n_clusters}, is outside [2, n_samples].")
            continue

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        # Define the silhouettes by the scores in the sample (range between -1 and 1)
        ax1.set_xlim([np.min(sample_silhouette_values), np.max(sample_silhouette_values)])

        # The (n_clusters + 1) * 10 is for a blank space between silhouettes (for clear demarcation)
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        y_lower = 10

        for i_, i in enumerate(np.unique(cluster_labels)):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i_) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("Silhouette Plot for the Various Clusters.")
        ax1.set_xlabel("Silhouette Coefficient Values")
        ax1.set_ylabel("Cluster Label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        # Add a clean set of xtick labels
        ax1.set_xticks(np.linspace(np.min(sample_silhouette_values),
                                   np.max(sample_silhouette_values),
                                   10).round(1))

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

        # Use PCA to visualize clusters
        pca = PCA(n_components=2)
        X_show = pca.fit_transform(X)
        ax2.scatter(X_show[:, 0], X_show[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        fitted_clusterer_labels = sorted(np.unique(fitted_clusterer.labels_))

        # Getting cluster centers from the data passed in
        if hasattr(fitted_clusterer, 'cluster_centers_'):
            centers = fitted_clusterer.cluster_centers_
        else:
            centers = np.array([X[fitted_clusterer.labels_ == i].mean(axis=0)
                                for i in fitted_clusterer_labels])

        centers_show = pca.transform(centers)

        # Draw white circles at cluster centers
        ax2.scatter(centers_show[:, 0], centers_show[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in zip(fitted_clusterer_labels, centers_show):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("PCA Visualization of Clustered Data.")
        ax2.set_xlabel("1st Principle Component")
        ax2.set_ylabel("2nd Principle Component")

        plt.suptitle((f"Silhouette Analysis with {params}"), fontsize=14, fontweight='bold')

    plt.show()


# Clustering
class ClusterComparison(object):
    def __init__(self, n_jobs=6, cl_algos_params=None):
        '''
        Get information about different clustering algorithms on data. Possible algorithms are 'AffinityProp', 'Agglomerative', 'Birch', 'DBSCAN',
        'Agglomerative_features', 'KMeans_minibatch', 'MeanShift', and 'Spectral'. An example use could be:

            cluster_params = {'KMeans_minibatch': {'n_clusters': [4, 8, 12]},
                  'DBSCAN': {'metric': ['cosine', 'euclidean', 'l1']},
                  'Spectral': {'n_clusters': [4, 8, 12]}}


        :param n_jobs:
        :param cl_algos_params:
        '''
        self.n_jobs = n_jobs

        if cl_algos_params is None:
            cl_algos_params = {'KMeans_minibatch': {'n_clusters': [4, 10]}}

        self.cl_algos_params = cl_algos_params

        self.cl_algos_all = {'AffinityProp': AffinityPropagation,
                           'Agglomerative': AgglomerativeClustering,
                           'Birch': Birch,
                           'DBSCAN': DBSCAN,
                           'Agglomerative_features': FeatureAgglomeration,
                           'KMeans_minibatch': MiniBatchKMeans,
                           'MeanShift': MeanShift,
                           'Spectral': SpectralClustering,
                           'OPTICS': OPTICS}

    def fit(self, X_df):

        self.X = scale(X_df.values.astype(np.float))
        self.cl_algos = {}
        self.df_clusters = pd.DataFrame(index=X_df.index)

        for algo_name in self.cl_algos_params.keys():

            print(f'\n**Training {algo_name} ...**')

            pgrid = ParameterGrid(self.cl_algos_params[algo_name])

            for i, params in enumerate(pgrid):

                print(f'\t**Using parameter setting {i+1} of {len(pgrid)} ...**')
                if algo_name in ['DBSCAN', 'MeanShift', 'Spectral', 'OPTICS']:
                    params['n_jobs'] = self.n_jobs

                cl_algo = self.cl_algos_all[algo_name](**params)
                clusters = cl_algo.fit_predict(self.X)

                self.cl_algos[f'{algo_name}_{i}'] = cl_algo
                self.df_clusters[f'clusters_{algo_name}_{i}'] = clusters

    def silhouette_plots(self, algo, report_on=('n_clusters')):

        algos = [key for key in self.cl_algos.keys() if algo in key]
        clusterers = [self.cl_algos[x] for x in algos]

        clusterers_labels_ = [self.df_clusters[f'clusters_{algo}'].values for algo in algos]

        silhouette_plots(self.X, fitted_clusterers=clusterers, report_on=report_on, clusterers_labels=clusterers_labels_)


def live_plot(data_dict, figsize=(12, 8), x_label='iter', y_label='value', title='', axis2=None, axis2_color='red', axis2_label=None):
    '''
    Rebuild plot dynamically given new data. For now, usage (assuming numpy and defaultdict imported):

    data = defaultdict(list)

    for i in range(100):
        data['label1'].append(np.sin(i/10) + 0.1*np.random.random())
        data['label2'].append(np.random.random()+0.3)
        live_plot(data)

    data_dict :: dict
        The dictionary of labels (keys) and lists of values (values) for those labels.

    '''

    clear_output(wait=True)
    fig, axis = plt.subplots()

    fig.set_size_inches(figsize[0], figsize[1])

    for label, data in data_dict.items():

        if label == axis2:

            if axis2_label is not None:
                label = axis2_label

            axis2 = axis.twinx()
            axis2.set_ylabel(label, color=axis2_color)
            axis2.tick_params(axis='y', colors=axis2_color)
            axis2.plot(data, label=label, color=axis2_color)

        else:
            axis.plot(data, label=label)

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.legend(loc='center left')

    plt.title(title)
    plt.grid(True)
    plt.show();
