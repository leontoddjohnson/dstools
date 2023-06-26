from gensim import models, similarities, matutils
from keras import layers
from ._preprocessing import vectorize, mytokenizer, sequence
import keras
import gensim.downloader as api
import numpy as np
import pickle as pkl
import random
import sys
import glob


class Dirichlet(object):

    def __init__(self, X, X_vec=None, vec=None, lda=None):
        '''
        Set up a class for LDA.
        :param X: array-like, List of documents, in rows.
        :param X_vec: sparse.matrix, Vectorized matrix of doc-term space (optional)
        :param vec: sklearn.CountVectorizer or sklearn.TfIdfVectorizer, vectorizer if already trained (optional)
        :param lda: gensim. LDA model, if already trained (optional)
        '''
        self.X = X
        self.X_vec = X_vec

        if X_vec:
            self.X_vec_T = self.X_vec.T
            self.corpus = matutils.Sparse2Corpus(self.X_vec_T)

        self.vec = vec

        if vec:
            self.id2word = dict((v, k) for k, v in self.vec.vocabulary_.items())

        self.lda = lda

    def vectorize(self, vecargs=None, CountV_args=None, ngram_range=(1, 2)):
        '''
        Use CountVectorizer to vectorize the corpus. LDA doesn't make much sense if you use TF-IDF.
        :param vecargs: (dict, optional) extra arguments to send to the pystuff.proprocess.vectorize function
        :param CountV_args: (dict, optional) arguments to send to the sklearn.CountVectorizer function.
        :param ngram_range: (tuple, optional) By default, we keep from individual values up to 2-grams, i.e., (1, 2)
        :return: None. instantiates the corpus and id2word dictionary.
        '''

        if not CountV_args:
            CountV_args = {}

        CountV_args_ = {**CountV_args, **{'ngram_range': ngram_range}}

        if not vecargs:
            vecargs = {}

        self.X_vec, self.vec = vectorize(self.X, use_tfidf=False, CountV_args=CountV_args_, **vecargs)
        self.terms = self.vec.get_feature_names()
        self.X_vec_T = self.X_vec.T
        self.corpus = matutils.Sparse2Corpus(self.X_vec_T)
        self.id2word = dict((v, k) for k, v in self.vec.vocabulary_.items())

    def fit(self, num_topics=3, passes=20, ldaargs=None, print_topics=False, num_words=10):
        '''
        AFTER the data is vectorized, fit an gensim.models.LdaModel to the corpus.
        :param num_topics: (int, optional) Number of topics for algorithm to calculate distributions for.
        :param passes: (int, optional) Number of passes through the corpus during training, i.e., how many times we
        use the Bayesian update process to approximate our Dirichlet distributions. (Update this parameter,
        don't add to the 'ldaargs' parameters)
        :param ldaargs: (dict, optional) Arguments to pass to LdaModel (other than corpus, num_topics, id2word, and
        passes).
        :param print_topics: (bool, optional) Whether or not to print the top words for each topic.
        :param num_words: (int, optional) Number of words to print probabilities for each topic.
        :return: (prints) if print_topics, prints the top words for each topic.
        '''
        if not ldaargs:
            ldaargs = {}

        self.lda = models.LdaModel(corpus=self.corpus, num_topics=num_topics,
                                   id2word=self.id2word, passes=passes, **ldaargs)

        # Transform our document-term space into document-topic space
        self.lda_corpus = self.lda[self.corpus]
        self.lda_docs = [doc for doc in self.lda_corpus]

        if print_topics:
            return self.lda.print_topics(num_words=num_words)

    def getsims(self, doc=0, num_show=3):
        '''
        Get the similarities for each document, given the document to check similarities for. Shows in order of
        'closeness' and uses cosine similarity = (x1 * x2) / (||x1||*||x2||). Closer to 1 is more similar,
        closer to 0 is less similar.
        :param doc: (int, optional) Document index to check similarites. Must be one of the documents (index) given
        in X.
        :param num_show: (int, optional) Number of documents to show similarities for.
        :return: (list) List of tuples with [0] docid, [1] cosine similarity.
        '''
        sim_index = similarities.MatrixSimilarity(self.lda_docs, num_features=self.X.shape[0])
        sims = sorted(enumerate(sim_index[self.lda_docs[doc]]), key=lambda item: -item[1])

        for sim_doc_id, sim_score in sims[0:num_show]:
            print("Document ID: " + str(sim_doc_id))
            print("Score: " + str(sim_score))
            print("Document: " + self.X[sim_doc_id])
            print('\n')

        return sims


class W2V_pretrained(object):

    def __init__(self, word_vectors=None, download=False, word_vector_library='word2vec-google-news-300'):
        '''
        Download or use preloaded word vectors using Gensim's installer.

        Note: Right now, only have functionality to download; might eventually use this for other work (e.g., transfer learning).

        Parameters
        ----------
        word_vectors : optional
            Preloaded word vectors
        download : bool
            Download from the `word_vector_library`, recognizable by gensim.
        word_vector_library : str
            gensim word vector library
        '''
        self.word_vectors = word_vectors

        if download:
            self.word_vectors = api.load(word_vector_library)
            with open('./word_vectors/' + word_vector_library + '.pkl', 'wb') as f:
                pkl.dump(self.word_vectors, f)

            print('Word Vectors downloaded and pickled into "./word_vectors/"')


class TextGen(object):

    def __init__(self, seed, kind='word', tokens_unique=None, token_indices=None, index_tokens=None, model=None,
                 lookback=10, step=1):
        '''
        This text generator only needs the original text (document) to base the generator off of. This should be one
        string.
        :param text: (str) The generator's seed.
        :param kind: (str) Either 'word' or 'char'.
        :param tokens_unique: (list) If using a generator, you need to have a list of all the unique words in the
        corpus.
        '''

        self.seed, _ = mytokenizer(seed, remove_stops=False, stem=False, keep_puncs='!?.', tokensonly=False)
        self.kind = kind
        self.lookback = lookback
        self.step = step
        self.tokens_unique = tokens_unique
        self.token_indices = token_indices
        self.index_tokens = index_tokens
        self.model = model

        if self.kind == 'word':
            self.tokens = [word for word in self.seed.split() if word not in ['!', '?', '.']]
        elif self.kind == 'char':
            self.tokens = list(self.seed)

    def generator(self, batch_size=128, dirname='./sacredtexts'):

        globs = glob.glob(dirname + '/**/*.txt', recursive=True)

        for fname in globs:

            with open(fname, encoding='utf-8', errors='ignore') as f:
                text = f.read()

            text_clean, _ = mytokenizer(text, remove_stops=False, stem=False, keep_puncs='!?.', tokensonly=False)

            if self.kind == 'word':
                tokens = [word for word in text_clean.split() if word not in ['!', '?', '.']]
            elif self.kind == 'char':
                tokens = list(text_clean)
            else:
                tokens = []

            x, y = sequence(tokens, self.tokens_unique, self.token_indices, self.lookback, self.step)

            size = y.shape[0]
            num_batches = size // batch_size

            for i in range(0, batch_size * num_batches, batch_size):
                yield x[i: i+batch_size], y[i: i+batch_size]

    def gentext_diag(self, dirname='./sacredtexts', epochs=30, steps_per_epoch=200, genlength=30, maxbatch=1000,
                     epoch_keep=1, temp_keep=1):

        model = keras.models.Sequential()

        model.add(layers.TimeDistributed(layers.Dense(28),
                                         input_shape=(self.lookback, len(self.tokens_unique))))
        # model.add(layers.LeakyReLU(alpha=.001))
        model.add(layers.CuDNNLSTM(64, input_shape=(self.lookback, len(self.tokens_unique))))
        model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
        model.add(layers.Dense(len(self.tokens_unique), activation='softmax'))

        optimizer = keras.optimizers.Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # When generating, temperature = 0.5 seems to work best.
        def sample(preds, temperature):
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)

            return np.argmax(probas)

        for epoch in range(1, epochs):
            print('epoch', epoch)

            # Fit the model for 1 epoch on the available data
            model.fit_generator(self.generator(dirname=dirname, batch_size=maxbatch),
                                steps_per_epoch=steps_per_epoch, epochs=1)

            # Select a text seed at random
            start_index = random.randint(0, len(self.tokens) - self.lookback - 1)

            generated_text = self.tokens[start_index: start_index + self.lookback]

            if self.kind == 'char':
                print('--- Generating with seed: "' + ''.join(generated_text) + '"')
            else:
                print('--- Generating with seed: "' + ' '.join(generated_text) + '"')

            for temperature in [0.5, 1.0]:
                print('------ temperature:', temperature)

                if self.kind == 'char':
                    sys.stdout.write(''.join(generated_text))
                else:
                    sys.stdout.write(' '.join(generated_text))

                for i in range(genlength):
                    sampled = np.zeros((1, self.lookback, len(self.tokens_unique)))
                    for t, token in enumerate(generated_text):
                        sampled[0, t, self.token_indices[token]] = 1.

                    preds = model.predict(sampled, verbose=0)[0]
                    next_index = sample(preds, temperature)
                    next_token = self.index_tokens[next_index]

                    generated_text.append(next_token)
                    generated_text = generated_text[1:]

                    if self.kind == 'char':
                        sys.stdout.write(next_token)
                    else:
                        sys.stdout.write(' ' + next_token)

                    sys.stdout.flush()

                if epoch == epoch_keep and temperature == temp_keep:
                    self.model = model

                print()

    def trainmodel(self, X=None, y=None, fit_args=None, use_generator=False, generator=None):

        # Copy paste this from the diag above.
        model = keras.models.Sequential()

        model.add(layers.TimeDistributed(layers.Dense(28),
                                         input_shape=(self.lookback, len(self.tokens_unique))))
        # model.add(layers.LeakyReLU(alpha=.001))
        model.add(layers.CuDNNLSTM(64, input_shape=(self.lookback, len(self.tokens_unique))))
        model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
        model.add(layers.Dense(len(self.tokens_unique), activation='softmax'))

        optimizer = keras.optimizers.Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        if use_generator:
            if not generator:
                generator = self.generator

            model.fit_generator(generator, **fit_args)

        else:
            model.fit(x=X, y=y, **fit_args)

        self.model = model

    def gentext(self, seed, genlength, temperature):

        generated_text_full = ' ... '

        def sample(preds, temperature):
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)

            return np.argmax(probas)

        start_index = random.randint(0, len(seed) - self.lookback - 1)
        generated_text = seed[start_index: start_index + self.lookback]

        for i in range(genlength):
            sampled = np.zeros((1, self.lookback, len(self.tokens_unique)))
            for t, token in enumerate(generated_text):
                sampled[0, t, self.token_indices[token]] = 1.

            preds = self.model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_token = self.index_tokens[next_index]

            generated_text_full = generated_text_full + next_token
            generated_text = generated_text + next_token
            generated_text = generated_text[1:]

        return generated_text_full