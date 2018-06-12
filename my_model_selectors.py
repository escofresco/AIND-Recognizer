import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    References:
        http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
        https://www.immagic.com/eLibrary/ARCHIVES/GENERAL/WIKIPEDI/W120607B.pdf
        http://www.statisticshowto.com/bayesian-information-criterion/
        https://stats.stackexchange.com/questions/90769/using-bic-to-estimate-the-number-of-k-in-kmeans

    Bayesian information criteria:
        BIC = -2 * logL + p * logN
        L: Likelihood of a "fitted" model
        p: Number of free parameters (model complexity); penalizes complex models
        N: Number of data points (size of data set)
        -2logL: Decreases with more parameters (p)
        plogN: Increases with p (complexity)
        low BIC: good modelgood model
        high BIC: bad model
    """
    def bic(self, nth_component):
        """ Find the BIC score
        :return: model, score (tuple)
        """
        model = self.base_model(nth_component)
        logL = model.score(self.X, self.lengths)
        N = len(self.X)
        logN = np.log(N)
        d = model.n_features
        p = nth_component**2 + 2*nth_component*d - 1

        BIC = -2*logL + p*logN
        return model, BIC

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None # Worst outcome is no model
        best_score = float("Inf") # Worst score possible is infinity
        try:
            for nth_component in range(self.min_n_components, self.max_n_components+1):
                # Search for BIC score between min and max components
                cur_model, cur_score = self.bic(nth_component)
                if cur_score < best_score:
                    best_model, best_score = cur_model, cur_score # Update globals with current best
                return best_model
        except:
            # Handle failure by returning default (taken from https://github.com/osanseviero/AIND/blob/master/Project4-ASLRecognizer/my_model_selectors.py)
            return self.base_model(self.n_constant)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    References:
        Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
        Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
        https://machinelearnings.co/sign-language-recognition-with-hmms-504b86a2acde

    Discriminative Information Criterion:
        DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
        log(P(X(i))): logL
        i: word to compare (self.this_word)
        1/(M-1)*SUM(log(P(X(all but i)))): mean(words excluding i)
    '''
    def dic(self, nth_component):
        """ Find DIC score
        :return: model, score (tuple)
        """
        model = self.base_model(nth_component)
        logL = model.score(self.X, self.lengths)
        # for word, (X, lengths) in self.hwords.items():
        #     # Iterate over words of length X
        #     if word != self.this_word:
        #         # Current word is not i ("all but i")

        # list comprehension compiling log(P(X(all but i)))
        scores = [model.score(X, lengths)
                    for word, (X, lengths) in self.hwords.items()
                    if word != self.this_word
                ]
        DIC = model.score(self.X, self.lengths) - np.mean(scores) # log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
        return model, DIC

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None # Worst outcome is no model
        best_score = float("-Inf") # Worst possible score is -infinity
        try:
            for nth_component in range(self.min_n_components, self.max_n_components+1):
                # Search for DIC score within min and max component constraints
                model, score = self.dic(nth_component)
                if score > best_score:
                    best_model, best_score = model, score # Update with current best
            return best_model
        except:
            # Handle failure by returning a default best model
            return self.base_model(self.n_constant)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    References:
        https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
    '''
    def cv(self, nth_component):
        """ Find cross validation score

        :return: model, score (tuple)
        """
        folds = min(len(self.sequences), 3)
        kfold = KFold(n_splits=folds) # Sklearn k-fold
        scores = []
        for train, test in kfold.split(self.sequences):
            # Iterate over kfold components
            self.X, self.lengths = combine_sequences(train, self.sequences) # Training data
            X_test, lengths_test = combine_sequences(test, self.sequences) # Testing data
            model = self.base_model(nth_component)
            scores.append(model.score(X_test, lengths_test))
        return model, np.mean(scores)

    def select(self):
        """
        :return: GaussianHMM
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None # Worst case is no model
        best_score = float("-Inf") # Worst possible score is infinity
        try:
            for nth_component in range(self.min_n_components, self.max_n_components+1):
                # Iterate over components within min max range
                model, score = self.cv(nth_component)
                if score > best_score:
                    best_model, best_score = model, score # Update globals with current best
            return best_model
        except:
            # Handle failure by return a default best model
            return self.base_model(self.n_constant)
