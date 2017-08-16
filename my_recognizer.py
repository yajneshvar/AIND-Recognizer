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
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for word_id, XLength in test_set.get_all_Xlengths().items():
        prob_dict = {}
        sequence, length = XLength
        for wd, model in models.items():
            try:
                prob_dict[wd] = model.score(sequence, length)
            except:
                prob_dict[wd] = float('-inf')
                continue
        probabilities.append(prob_dict)
        best_guess = max(prob_dict.items(),key=lambda x: x[1])
        guesses.append(best_guess[0])
    return probabilities, guesses
