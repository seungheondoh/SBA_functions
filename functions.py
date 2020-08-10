import os
import json
from six import string_types, integer_types
from numpy import dot, float32 as REAL, array, ndarray, sum as np_sum, prod

from gensim import matutils
from konlpy.tag import Komoran
import yake

var_path = os.path.abspath(os.path.dirname('./static/')) + "/user_dict.txt"
pos_tagger = Komoran(userdic=var_path)
save_item = ('NNG','NNP')

def keyword_extractor(sentence, topn=10):
    """Extract top-N keyword similar words.
    Parameters
    ----------
    sentence : str
        Sentence, Paragraph, Input Query.
    topn : int
        Number of top-N relevent keywords to return

    Returns
    -------
    list of extract keywords
        When `topn` is int, a sequence of extract keyword is returned.
    """
    kw_extractor = yake.KeywordExtractor(top=topn)
    keywords = kw_extractor.extract_keywords(sentence)
    return [word for _,word in keywords]

def _tokenize(sentence):
    return ['/'.join(t) for t in pos_tagger.pos(sentence)]

def tokenizer(sentence):
    """Tokenize sentence
    Parameters
    ----------
    sentence : str
        Sentence, Paragraph, Input Query.
    Returns
    -------
    list of extract keywords
    """
    token_doc = _tokenize(sentence)
    return [token.split('/')[0] for token in token_doc if (token.split('/')[1] in save_item) and len(token.split('/')[0]) > 1]

def multiquery_retrieval(self, positive, restrict_vocab, topn=5):
    """Find the top-N most similar words.
    Parameters
    ----------
    self: model.wv
        Class of Gensim Word2vec model.wv 
    positive : list of str
        List of words that contribute positively.
    restrict_vocab : serach space
        List of Tag and Song
    topn : int
        Number of top-N similar words to return
    Returns
    -------
    list of (str, float) or numpy.array
        a sequence of retrieval tag and song
    """
    if positive is None:
        positive = []
    self.init_sims()

    if isinstance(positive, string_types):
        # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
        positive = [positive]
        
    positive = [word for word in positive if word in self.vocab]
    positive = [(word, 1.0) if isinstance(word, string_types + (ndarray,)) else word for word in positive]
    
    # compute the weighted average of all words
    all_words, mean = set(), []
    for word, weight in positive:
        if isinstance(word, ndarray):
            mean.append(weight * word)
        else:
            mean.append(weight * self.word_vec(word, use_norm=True))
            if word in self.vocab:
                all_words.add(self.vocab[word].index)
    if not mean:
        raise ValueError("아쉽게도, 검색어와 어울리는 노래를 찾을 수 없습니다. 사람들이 가장 좋아하는 노래를 검색할까요?")
        
    mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)
    limited = self.vectors_norm if restrict_vocab is None else self.vectors_norm[restrict_vocab]
    dists = dot(limited, mean)
    
    if not topn:
        return dists
    best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)

    if restrict_vocab:
        result = [(self.index2word[restrict_vocab[sim]], float(dists[sim])) for sim in best if restrict_vocab[sim] not in all_words]
    else:
        result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]

    return [fname for fname, _ in result[:topn]]