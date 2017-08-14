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
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
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

    def p_val(self, n_components):
        """
        Calculate the number of free parameters based off discussion below
        https://discussions.udacity.com/t/understanding-better-model-selection/232987/3
        """
        n, feature_length = self.X.shape
        return n_components**2 - 2*n_components*feature_length - 1


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

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def bic_score(self, n_components, logl):
        n, feature_len = self.X.shape
        return -2 * logl + self.p_val(n_components) * np.log(n)

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            score_list = []
            for n in range(self.min_n_components, self.max_n_components + 1):
                trained_model = self.base_model(n)
                if trained_model:
                    log_l = trained_model.score(self.X, self.lengths)
                    final_score = self.bic_score(n, log_l)
                    score_list.append((trained_model, final_score))
                if score_list:
                    best_model, best_score = max(score_list, key=lambda x: x[1])
                    return best_model
        except Exception as e:
            return None


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def compute_mean_logL_others(self, word, trained_model):
        def calc_score(t_wd, model):
            try:
                log_l = model.score(self.hwords[t_wd])
                return log_l
            except:
                return 0

        other_words = [otr_wd for otr_wd in self.words if otr_wd != word]
        score_otr_word = [calc_score(wd, trained_model) for wd in other_words]
        return np.mean(score_otr_word, axis=None)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model_score = []
        for n in range(self.min_n_components, self.max_n_components + 1):
            trained_model = self.base_model(n)
            if trained_model:
                try:
                    log_l_of_x = trained_model.score(self.X, self.lengths)
                    log_l_not_x = self.compute_mean_logL_others(self.this_word, trained_model)
                    dic_score = log_l_of_x - log_l_not_x
                    model_score.append((trained_model, dic_score))
                except:
                    model_score.append((None, 0))

        best_model = max(model_score, key=lambda x: x[1])
        return best_model[0]



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # first train all hmmm based on fold
        # then validate by scoring against test sample
        # select the highest avg log likelyhood model
        split_method = KFold()
        model_list = []
        for i in range(self.min_n_components, self.max_n_components + 1):
            log_llist = []
            trained_model = None
            try:
                if len(self.sequences) > 2:
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        x_train, len_train = combine_sequences(cv_train_idx, self.sequences)
                        self.X = x_train
                        self.lengths = len_train
                        x_test, len_test = combine_sequences(cv_test_idx, self.sequences)
                        trained_model = self.base_model(i)
                        log_l = trained_model.score(x_test, len_test)
                        log_llist.append(log_l)
                else:
                    trained_model = self.base_model(i)
                    log_l = trained_model.score(self.X, self.lengths)
            except:
                return None
            avg = 0
            if len(log_llist) > 0:
                avg = sum(log_llist, 0) / len(log_llist)
            model_list.append((trained_model, avg))
            best_model = max(model_list, key=lambda x: x[1])
            return best_model[0]
