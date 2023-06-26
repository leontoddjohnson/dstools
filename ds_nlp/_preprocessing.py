from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from . import __path__ as ROOT_PATH
from nltk.tokenize import word_tokenize, sent_tokenize
from copy import copy
from glob import glob
import numpy as np
import string
import re


def mytokenizer(text, remove_stops=True, stop_words='nltk', stemming='snowball', drop_punc=True,
                drop_digits=True, stem=True, otherpunc='', keep_puncs='', tokensonly=True):
    '''
    This is a custom tokenizer. We make tokens in this order:
        1) Remove URLs (this is unchangeable),
        2) Removing punctuation,
        3) Lowercase-ing,
        4) Removing stopwords (see the list in the data folder), ... It assumes the English language.
        5) Applying a stemmer.
    * If using Word2Vec, do not remove stopwords. According to a Kaggle site 'To train Word2Vec it is better not to
    remove stop words because the algorithm relies on the broader context of the sentence in order to produce
    high-quality word vectors'. You can try both -- see what happens.
    :param text: str, This is the text string to be cleaned and tokenized.
    :param remove_stops: bool, Just decide if you want to remove stop words or not. Default is True
    :param stop_words: str, in ('nltk', 'google', 'ranksnl'). The 'nltk' stopwords are the default NLTK ones,
    and the others come from www.ranks.nl (google, by way of). The Google list is by far the longest.
    :param stemming: str, in ('snowball', 'porter', or 'lancaster')
    :param drop_puncnums: bool, Self-explanatory. (drop the punctuation and the numbers)
    :param stem: bool, Self-explanatory.
    :param keep_puncs: (str), A string of punctuation characters you want to keep in the text.
    :return: (tuple), [0] the cleaned text on its own and [1] the tokenized string values. Note, the clean text will
    only have cleaned up the number sand punctuation.
    '''

    # with open(f'{ROOT_PATH[0]}/data/punc.txt', 'r') as f:
    #     otherpunc = f.readlines()
    #
    # otherpunc = ''.join(set([x.strip() for x in otherpunc]))
    punctuation = otherpunc + string.punctuation
    punctuation = ''.join([x for x in punctuation if x not in list(keep_puncs)])

    urlpat = '(https?:\/\/)(\s)*(www\.)?(\s)*(\w+\.)*([\w\-\s]+\/)*([\w\-]+)\/?(\??\w+\s*=\w+\&?)*'

    # Define substitution methods
    clean_text = copy(text)
    clean_text = re.sub(urlpat, '', clean_text)

    remove_punct = str.maketrans('', '', punctuation)

    if drop_punc:
        clean_text.replace('--', ' ')
        clean_text = clean_text.translate(remove_punct)

    if drop_digits:
        remove_digit = str.maketrans('', '', string.digits)
        clean_text = clean_text.translate(remove_digit)

    clean_text = clean_text.lower()
    clean_tokens = word_tokenize(clean_text)

    if remove_stops:
        if stop_words == 'google':
            with open(f'{ROOT_PATH[0]}/data/sw_google.txt', 'r') as f:
                stop_words = f.readlines()
                stop_words = [word.strip() for word in stop_words]

        elif stop_words == 'ranksnl':
            with open(f'{ROOT_PATH[0]}/data/sw_ranksnl.txt', 'r') as f:
                stop_words = f.readlines()
                stop_words = [word.strip() for word in stop_words]

        else:
            stop_words = stopwords.words('english')

        # Some of the lists of stopwords haven't removed the punctuation. :| (neutral face)
        stop_words = [word.translate(remove_punct) for word in stop_words]
        stop_words = set(stop_words)
        clean_tokens = [word for word in clean_tokens if word not in stop_words]

    stemmer = SnowballStemmer('english', ignore_stopwords=True)

    if stemming == 'porter':
        stemmer = PorterStemmer()

    if stemming == 'lancaster':
        stemmer = LancasterStemmer()

    if stem:
        clean_tokens = [stemmer.stem(y) for y in clean_tokens]

    if tokensonly:
        return clean_tokens
    else:
        return clean_text, clean_tokens


def vectorize(X, tokenizer=mytokenizer, use_tfidf=True, Tfidf_args=None, CountV_args=None):
    '''
    !RETURNS A TUPLE, DUDE!
    This is a custom vectorizer which does a TF-IDF transformation by default. It returns a scipy sparse matrix to be
    used with sklearn machine learning functions, and the vectorizer that was trained on it (using the parameters
    defined).
    :param X: array, These are the data to be transformed.
    :param tokenizer: func, Tokenizer to use. Defaults to the default of mytokenizer
    :param use_tfidf: bool, Run the TF-IDF transformation, or not.
    :param Tfidf_args: dict, Parameters and values to pass to the TfidfVectorizer function
    :param CountV_args: dict, Parameters and values to pass to the CountVectorizer function
    :return: tuple, [0] scipy.sparse.csr.csr_matrix, the transformed matrix in sparse format, [1] The vectorizer.
    '''

    if not Tfidf_args:
        Tfidf_args = dict()

    if not CountV_args:
        CountV_args = dict()

    vec = TfidfVectorizer(tokenizer=tokenizer, **Tfidf_args)

    if not use_tfidf:
        vec = CountVectorizer(tokenizer=tokenizer, **CountV_args)

    X_vec = vec.fit_transform(X)

    return X_vec, vec


def doc2sent(X_docs, remove_stops=False, stop_words='nltk', stemming='snowball', drop_puncnums=True, stem=True):
    '''
    This is mostly to prepare a list of documents for building a Word2Vec model. If you have data where each record
    is a piece of text (i.e., a document), use this function to output a new array where each record is a list of
    words of a sentence.
    :param X_docs: (np.array, pd.DataFrame) This is the original data, with each record a document text.
    :param remove_stops: (bool, option) Do you want to remove stop words? In some cases, it may work out best to keep
    this as True, but for our purposes, we kept it as False.
    :param stop_words: (str, optional) In ('nltk', 'short', 'med', or 'long'). The 'nltk' stopwords are the default
    NLTK ones,
    and the others come from www.ranks.nl. short/med/long refers to the amount of stopwords in the list.
    :param stemming: (str, optional) In ('snowball', 'porter', or 'lancaster')
    :param drop_puncnums: (bool, optional) Drop the punctuation and the numbers
    :param stem: (bool, optional) Self-explanatory.
    :return: (list) A list of all the sentences for each document in tokenized format.
    '''
    X_sent = X_docs.apply(sent_tokenize)

    # Take list of lists, compile the union.
    X_sent = [item for sublist in X_sent for item in sublist]

    # Word2Vec wants sentences as lists of words
    X_sent_tokens = [mytokenizer(text, remove_stops=remove_stops,
                                 stop_words=stop_words, stemming=stemming,
                                 drop_puncnums=drop_puncnums, stem=stem) for text in X_sent]

    return X_sent_tokens


def generator(data, lookback, delay, min_index=0, max_index=None, shuffle=False, batch_size=128, step=6):
    '''
    This is adopted from Francois Challet in his guide to Keras.
    :param data:
    :param lookback:
    :param delay:
    :param min_index:
    :param max_index:
    :param shuffle:
    :param batch_size:
    :param step:
    :return:
    '''
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets


def sequence(tokens, tokens_unique=None, token_indices=None, maxlen=10, step=1):
    '''
    This is adopted from the Keras code provided by Francois Challet in his guide to Keras. Originally,
    it was meant to be character-based, but this is adopted to be either.
    :param maxlen: (int) This is the maximum length of the X sequences in 'kind' (e.g., if kind='words',
    then this is the maximum value for "pre-sequences" like ['hello', 'im'] --> 'leon' would have maxlen of 2 if
    we're using kind='words')
    :param step: (int) Overlap. For each of the "pre-sequences", how much should the next overlap with the
    previous. In other words, how many of the first words of the previous should *not* be in the next?
    :return: None. This sets up x and y for the model to train on
    '''

    sentences = []
    next_tokens = []
    maxlen = maxlen

    if not token_indices:
        token_indices = dict((t, i) for i, t in enumerate(tokens_unique))

    # Here we make our sequences which correspond to one another
    for i in range(0, len(tokens) - maxlen, step):
        sentences.append(tokens[i: i + maxlen])
        next_tokens.append(tokens[i + maxlen])

    # Basically, we want to have our input data (x) in tensor format:

    # sentence_1:   [  [token_1_vector] [token_2_vector] ... [token_t_vector] ... [token_k_vector]  ]
    # sentence_2:   [  [token_1_vector] [token_2_vector] ... [token_t_vector] ... [token_k_vector]  ]
    #    ...
    # sentence_i:   [  [token_1_vector] [token_2_vector] ... [token_t_vector] ... [token_k_vector]  ]
    #    ...
    # sentence_n:   [  [token_1_vector] [token_2_vector] ... [token_t_vector] ... [token_k_vector]  ]

    # NOTE: sentence_t is just the sentence that starts with the `step`th token of sentence_t-1 of length k.
    # Where each sentence i has k = maxlen tokens, and [token_t_vector] is a one hot vector of dimension
    # len(tokens_unique), with a 1 (True) if the token_t is that token, and 0 (False) otherwise. We want
    # our data like this because in the end, we want a probability distribution among the possible token values.

    x = np.zeros((len(sentences), maxlen, len(tokens_unique)), dtype=np.bool)
    y = np.zeros((len(sentences), len(tokens_unique)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, token in enumerate(sentence):
            x[i, t, token_indices[token]] = 1
        y[i, token_indices[next_tokens[i]]] = 1

    return x, y


class IterSentences(object):
    def __init__(self, dirname='./blah', min_sent_char=10, maxtexts=None):
        '''
        This makes an iterator which goes through all of the *.txt files in a directory, removes punctuation (except
        for periods, exclamation marks, and question marks), lowercases, and yields a list of the words in each
        sentence of each text. For example, if there are two files in a directory, and a subdirectory with another
        text file in it, this will yield lists of the words in the first file, then the next file, the next file,
        and then the words in the file in the subdirectory. I.e., this function works recursively.
        :param dirname: (str) The directory with all the texts in it.
        :param min_sent_char: (int) The minimum number of characters in a 'sentence'. At the time of making this,
        we have a few 'sentences' which end up being just periods or question marks, because of the way the data was
        gathered.
        :param maxtexts: (int) The maximum number of texts to go through. By default, this will go through all of the
        texts in the directory. It's best only to use this if you're diagnosing.
        '''
        self.globs = glob(dirname + '/**/*.txt', recursive=True)
        if maxtexts:
            self.maxtexts = maxtexts
        else:
            self.maxtexts = len(self.globs)
        self.min_sent_char = min_sent_char

    def __iter__(self):
        for fname in self.globs[:self.maxtexts]:
            with open(fname, encoding='utf-8', errors='ignore') as f:
                text = f.read()
            text_clean, _ = mytokenizer(text, remove_stops=False, stem=False, keep_puncs='!?.', tokensonly=False)
            sentences = [sent for sent in sent_tokenize(text_clean) if len(sent) > self.min_sent_char]
            for sentence in sentences:
                yield [word for word in sentence.split() if word not in ['!', '?', '.']]