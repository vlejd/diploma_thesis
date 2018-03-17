import os, io
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

SENTEVAL_DATA_BASE = '/home/vlejd/Documents/school/diplomka/SentEval/data/senteval_data'
class Dataset(object):
    def __init__(self, seed=1111):
        self.seed = seed
        self.positives = self.load_positives()
        self.negatives = self.load_negatives()
        self.samples = self.positives + self.negatives
        self.labels = np.array([1] * len(self.positives) + [0] * len(self.negatives))
        self.n_samples = len(self.samples)
        self.reshufle(seed)

    def reshufle(self, random_state=None):
        shuffle = StratifiedShuffleSplit(10, random_state=random_state)
        generator = shuffle.split(self.samples, self.labels)
        self.train_id, self.test_id = next(iter(generator))

    def train_samples(self):
        return np.array(self.samples)[self.train_id]

    def train_labels(self):
        return self.labels[self.train_id]

    def test_samples(self):
        return np.array(self.samples)[self.test_id]

    def test_labels(self):
        return self.labels[self.test_id]

    def loadFile(self, fpath):
        with io.open(fpath, 'r', encoding='latin-1') as f:
            return [line.split() for line in f.read().splitlines()]

    def baseline(self):
        mean = np.mean(self.labels)
        return max(mean, 1-mean)

    def name(self):
        return self.__class__.__name__
    
class CRDataset(Dataset):
    def load_positives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'CR/custrev.pos'))

    def load_negatives(self): 
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'CR/custrev.neg'))

class MRDataset(Dataset):
    def load_positives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'MR/rt-polarity.pos'))

    def load_negatives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'MR/rt-polarity.neg'))

class SUBJDataset(Dataset):
    def load_positives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'SUBJ/subj.objective'))

    def load_negatives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'SUBJ/subj.subjective'))

class MPQADataset(Dataset):
    def load_positives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'MPQA/mpqa.pos'))

    def load_negatives(self):
        return self.loadFile(os.path.join(SENTEVAL_DATA_BASE, 'MPQA/mpqa.neg'))       

class DebugDataset(CRDataset):
    def __init__(self, seed=1111):
        super().__init__(seed)

        self.positives = self.positives[:100]
        self.negatives = self.negatives[:100]
        self.samples = self.positives + self.negatives
        self.labels = np.array([1] * len(self.positives) + [0] * len(self.negatives))
        self.n_samples = len(self.samples)
        self.reshufle(seed)

ALL_DATASETS = [CRDataset(), MRDataset(), SUBJDataset(), MPQADataset()]