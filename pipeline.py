from utils import *
import pandas as pd


def load():
    fname = "data/train.csv"
    train = pd.read_csv(fname, header=0)
    test = "data/test.csv"
    test = pd.read_csv(fname, header=0)
    return train, test


class Model():

    def train(data):
        self.work_data = data.copy()
        work_data.question1 = data.question1.apply(normalize)
        work_data.question2 = data.question2.apply(normalize)
        pom = pd.concat([work_data.question1, work_data.question2])
        self.self.dic = Dictionary(pom)
        # self.dic.filter_extremes(no_below=2, no_above=)
        vectorized_data = work_data.copy()
        vectorized_data.question1 = work_data.question1.apply(lambda x: self.dic.doc2bow(x))
        vectorized_data.question2 = work_data.question2.apply(lambda x: self.dic.doc2bow(x))
        q1 = gensim.matutils.corpus2csc(vectorized_data.question1).transpose()
        q2 = gensim.matutils.corpus2csc(vectorized_data.question2).transpose()
        q1._shape = (vectorized_data.shape[0], len(self.dic.dfs))
        q2._shape = (vectorized_data.shape[0], len(self.dic.dfs))

        X = q1.multiply(q2)
        X[X > 1] = 1
        Y = vectorized_data.is_duplicate
        kf = KFold(n_splits=2)
        clf = LogisticRegression(class_weight=None, max_iter=100, verbose=100, warm_start=False, n_jobs=4)
        for train_index, test_index in kf.split(X):
            clf.fit(X[train_index],Y[train_index])
            probs = clf.predict_proba(X[test_index])
            print(loss(Y[test_index], probs))

def prepare(data):


if __name__ == "__main__":
    pass
