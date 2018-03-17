from sklearn.linear_model import LogisticRegression
import gensim
import numpy as np
from term_weights import SupervisedTermWeightingWTransformer, UnsupervisedTfidfTransformer

class SimpleModel(object):
    
    SCHEMES = ('None', 'tfidf', 'tfchi2', 'tfig', 'tfgr', 'tfor', 'tfrf', 'None')
    TRANSFORMERS = { 
        'tfidf': UnsupervisedTfidfTransformer(norm=None),
        'tfchi2': SupervisedTermWeightingWTransformer(scheme='tfchi2'),
        'tfig': SupervisedTermWeightingWTransformer(scheme='tfig'),
        'tfgr': SupervisedTermWeightingWTransformer(scheme='tfgr'),
        'tfor': SupervisedTermWeightingWTransformer(scheme='tfor'),
        'tfrf': SupervisedTermWeightingWTransformer(scheme='tfrf'),
        'None': None,
        None: None
    }

    def __init__(self, cls=None, weights=None, w=None, use_svd=True):
        self.cls = cls
        if self.cls is None:
            self.cls = LogisticRegression()
        self.weight_model = self.TRANSFORMERS[weights] 
        self.w = w
        self.use_svd = use_svd
        


    def fit(self, X, Y):
        self.dictionary = gensim.corpora.Dictionary(X)
        self.num_terms = len(self.dictionary.dfs)
        bow = list(map(self.dictionary.doc2bow, X))
        bow = gensim.matutils.corpus2csc(bow, num_terms=self.num_terms).T
        if self.weight_model is not None:
            self.weight_model.fit(bow, Y)
            bow = self.weight_model.transform(bow)
        if self.w is None:
            self.w = np.ones(self.num_terms)
        if self.use_svd:
            bow = gensim.matutils.Sparse2Corpus(bow.T)
            self.lsi = gensim.models.LsiModel(bow, id2word=self.dictionary)
            self.corpus = gensim.matutils.corpus2dense(self.lsi[bow], self.lsi.num_topics).T
        else:
            self.corpus = bow.multiply(self.w)
            
        self.cls.fit(self.corpus, Y)
    
    def dw(self, X, Y):
        _ = self.predict(X)
        self.d_embedding = self.cls.dx(self.embedding, Y)
        u = self.lsi.projection.u
        res = self.d_embedding.dot(u.T)
        dw = model.bow.multiply(res.T).sum(axis=1).T.A1
        return dw
        
    
    def predict(self, X):
        bow = list(map(self.dictionary.doc2bow, X))
        bow = gensim.matutils.corpus2csc(bow, num_terms=self.num_terms).T

        if self.weight_model is not None:
            bow = self.weight_model.transform(bow)

        if self.use_svd:
            bow = bow.multiply(self.w)
            bow = gensim.matutils.Sparse2Corpus(bow.T)
            self.embedding = gensim.matutils.corpus2dense(self.lsi[bow], self.lsi.num_topics).T
        else:
            self.embedding = bow.multiply(self.w)            
        Yhat = self.cls.predict(self.embedding)
        return Yhat
    
    def score(self, X, Y):
        Yhat = self.predict(X)
        return 1-((Yhat-Y)**2).mean()
    
    def update(self, X, Y):
        pass