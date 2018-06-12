import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]

    References:
        https://stackoverflow.com/questions/10458437/what-is-the-difference-between-dict-items-and-dict-iteritems
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for cur_X, cur_lengths in test_set.get_all_Xlengths().values():
        # Iterate over SinglesData
        best_score = float('-Inf') # Worst possible score is infinity
        best_guess = None # Worst case is there's no best word
        likelihoods = {} # Track logL for each word
        for word, model in models.items():
            # Iterate over all models
            try:
                logL = model.score(cur_X, cur_lengths)
                likelihoods[word] = logL
                if logL > best_score:
                    # This must be the current best guess
                    best_guess, best_score = word, logL
            except:
                # Append default worst score (-Infinity) to likelihoods
                likelihoods[word] = float('-Inf')
        probabilities.append(likelihoods)
        guesses.append(best_guess)
    return probabilities, guesses
