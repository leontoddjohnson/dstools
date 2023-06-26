from sklearn.decomposition import NMF, TruncatedSVD
import numpy as np
from ._preprocessing import mytokenizer, vectorize
import pandas as pd


class TextDecomposition(object):

    def __init__(self, X, X_vec=None, vec=None, nmf=None, svd=None):
        '''
        Create a working class for use in NMF and SVD analysis.
        :param X: array-like, List of documents, in rows.
        :param X_vec: sparse.matrix, If you have a pre-vectorized matrix, here to save time.
        :param vec: vectorizer, If you've already pretrained a vectorizer (TF-IDF or CountVectorizer)
        :param nmf: sklearn.decomposition.NMF, If already pretrained.
        :param svd: sklearn.decomposition.TruncatedSVD, If already pretrained.
        '''
        self.X = X
        self.X_vec = X_vec
        self.vec = vec
        self.nmf = nmf
        self.svd = svd

    def vectorize(self, tokenizer=mytokenizer, use_tfidf=True, Tfidf_args=None, CountV_args=None):
        '''
        Vectorize the X (list of documents) if not done already.
        :param tokenizer: func, Tokenizer to use. Defaults to the default of mytokenizer
        :param use_tfidf: bool, Run the TF-IDF transformation, or not.
        :param Tfidf_args: dict, Parameters and values to pass to the TfidfVectorizer function
        :param CountV_args: dict, Parameters and values to pass to the CountVectorizer function
        :return: None, updates X_vec, vec, and terms
        '''

        if self.X_vec is None or self.vec is None:
            if not Tfidf_args:
                Tfidf_args_ = dict()
            else:
                Tfidf_args_ = Tfidf_args

            if not CountV_args:
                CountV_args_ = dict()
            else:
                CountV_args_ = CountV_args

            self.X_vec, self.vec = vectorize(self.X, tokenizer=tokenizer, use_tfidf=use_tfidf,
                                             Tfidf_args=Tfidf_args_, CountV_args=CountV_args_)

            self.terms = self.vec.get_feature_names()

        else:
            self.terms = self.vec.get_feature_names()
            print("Word matrix and vectorizer already set, updated/reset terms.")

    def fit_nmf(self, n=10, modelargs=None, re_pat=None):
        '''
        Fit an NMF model to the X_vec (vectorized doc-term matrix). By default, this is initialized with 'nndsvd'.
        :param n: int, Number of components
        :param modelargs: dict, other parameters to send to the sklearn.NMF model
        :return: None, updates the nmf_docspace and nmf_termspace.
        '''

        if self.nmf is None:
            if not modelargs:
                modelargs = {}

            self.nmf = NMF(n_components=n, init='nndsvd', **modelargs)

        if re_pat is not None:
            terms = pd.Series(self.terms)
            mask = terms.str.contains(re_pat, regex=True)
            self.terms = terms[mask]
            X_vec = self.X_vec[mask]

        else:
            X_vec = self.X_vec

        self.nmf_docspace = self.nmf.fit_transform(X_vec)
        self.nmf_termspace = self.nmf.components_

    def fit_svd(self, n=20, modelargs=None):
        '''
        Fit an SVD model to the X_vec.
        :param n: Number of components
        :param modelargs: dict, other parameters to send to sklearn.TruncatedSVD model
        :return: None, updates the docpsace, termspace, and sigma matrices
        '''

        if not modelargs:
            modelargs = {}

        self.svd = TruncatedSVD(n_components=n, **modelargs)
        self.svd_docspace = self.svd.fit_transform(self.X_vec)
        self.svd_termspace = self.svd.components_
        self.svd_sigma = self.svd.singular_values_

    def getclusters(self, model='nmf'):
        '''
        ! RETURNS TUPLE !
        Get a list of clusters for the docs and terms (i.e., which topics they should be assigned based on coefficients.
        :param model: 'nmf' or 'svd', Depending on which models have been fit already.
        :return:
        [0] nd.array, document clusters
        [1] nd.array, document cluster coefficients
        [2] nd.array, term clusters
        [3] nd.array, term cluster coefficients
        '''
        if model == 'nmf':
            docspace = self.nmf_docspace
            termspace = self.nmf_termspace

        elif model == 'svd':
            docspace = self.svd_docspace
            termspace = self.svd_termspace

        else:
            raise AttributeError("Only have capability for NMF and SVD factorization for now ...")

        doc_clusters = docspace.argmax(axis=1)
        term_clusters = termspace.argmax(axis=0)
        doc_clusters_coeffs = docspace.max(axis=1)
        term_clusters_coeffs = termspace.max(axis=0)

        return doc_clusters, doc_clusters_coeffs, term_clusters, term_clusters_coeffs

    def top_terms(self, model='nmf', topic=0, n=20):
        '''
        Given a model, print the words that are most associated with that topic by coefficient.
        :param model: 'nmf' or 'svd', which model to use.
        :param topic: int, which of the topics to print terms for.
        :param n: Number of words to print
        :return: None.
        '''
        termspace = self.nmf_termspace

        if model == 'svd':
            termspace = self.svd_termspace

        term_clusters = termspace.argmax(axis=0)
        term_topic_max_value = termspace.max(axis=0)

        self.term_topics = list(zip(self.terms, term_clusters, term_topic_max_value))

        return sorted([term for term in self.term_topics if term[1] == topic], key=lambda x: x[2], reverse=True)[:n]

    def top_docs(self, model='nmf', topic=0, n=20):
        '''
        Given a model, print the documents that are most associated with that topic, by coefficient.
        :param model: 'nmf' or 'svd', which model to use.
        :param topic: int, which of the topics to print terms for.
        :param n: Number of words to print
        :return: None.
        '''
        docspace = self.nmf_docspace

        if model == 'svd':
            docspace = self.svd_docspace

        doc_clusters = docspace.argmax(axis=1)
        doc_topic_max_value = docspace.max(axis=1)

        self.doc_topics = list(zip(range(len(self.X)), self.X, doc_clusters, doc_topic_max_value))

        for i, doc_topic in enumerate(
                sorted([doc_topic for doc_topic in self.doc_topics if doc_topic[2] == topic], key=lambda x: x[3],
                       reverse=True)[:n]):
            print('Topic: ', str(doc_topic[2]))
            print('Document ID: ' + str(doc_topic[0]))
            print('Score: ' + str(doc_topic[3]))
            print('Document: ' + doc_topic[1])
            print('\n')

    def term_vec(self, term='paradox', model='nmf', verbose=True):
        '''
        Given a term that is in the corpus for a given model, give the vector in the topic-termspace.
        :param term: (str) Term to get the topic-term vector for.
        :param model: (str, in ['nmf', 'svd']) The model you'd like to look in the topic-termspace.
        :param verbose: (bool) If True, print the argument maximum of the term vector.
        :return: (np.array) The term vector for the given term, and print the argument maximum (if verbose)
        '''
        if model == 'svd':
            termspace = self.svd_termspace
        else:
            termspace = self.nmf_termspace

        term_index = self.terms.index(term)
        term_vec = termspace[:, term_index]

        if verbose:
            print('Arg_max: ', np.argmax(term_vec))

        return term_vec