from sklearn.linear_model import LogisticRegression
import gensim
import numpy as np

class SimpleModel(object):
    def __init__(self, cls=None, use_tfidf=False, w=None, use_svd=True):
        self.cls = cls
        if self.cls is None:
            self.cls = LogisticRegression()
        self.use_tfidf = use_tfidf
        self.w = w
        self.use_svd = use_svd
        

    def fit(self, X, Y):
        self.dictionary = gensim.corpora.Dictionary(X)
        self.num_terms = len(self.dictionary.dfs)
        bow = list(map(self.dictionary.doc2bow, X))
        if self.use_tfidf:
            self.tfidf_model = gensim.models.TfidfModel(bow)
            bow = self.tfidf_model[bow]
        if self.w is None:
            self.w = np.ones(self.num_terms)
        if self.use_svd:
            bow = gensim.matutils.corpus2csc(bow, num_terms=self.num_terms)
            bow = gensim.matutils.Sparse2Corpus(bow.multiply(self.w.reshape(-1,1)))
            self.lsi = gensim.models.LsiModel(bow, id2word=self.dictionary)
            self.corpus = gensim.matutils.corpus2dense(self.lsi[bow], self.lsi.num_topics).T
        else:
            bow = gensim.matutils.corpus2csc(bow, num_terms=self.num_terms)
            self.corpus = bow.multiply(self.w.reshape(-1,1)).T
            
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
        if self.use_tfidf:
            bow = self.tfidf_model[bow]
        if self.use_svd:
            bow = gensim.matutils.corpus2csc(bow, num_terms=self.num_terms)
            self.bow = bow
            bow = bow.multiply(self.w.reshape(-1,1))
            bow = gensim.matutils.Sparse2Corpus(bow)
            self.embedding = gensim.matutils.corpus2dense(self.lsi[bow], self.lsi.num_topics).T
        else:
            bow = gensim.matutils.corpus2csc(bow, num_terms=self.num_terms)
            self.bow = bow
            self.embedding = bow.multiply(self.w.reshape(-1,1)).T            
        Yhat = self.cls.predict(self.embedding)
        return Yhat
    
    def score(self, X, Y):
        Yhat = self.predict(X)
        return 1-((Yhat-Y)**2).mean()
    
    def update(self, X, Y):
        pass