import io
import os
import numpy as np

SENTEVAL_DATA_BASE = '/home/vlejd/Documents/school/diplomka/SentEval/data/senteval_data'
class Dataset(object):
    def __init__(self, seed=1111):
        self.seed = seed
        self.load_positives()
        self.load_negatives()
        self.samples = self.positives + self.negatives
        self.labels = np.array([1] * len(self.positives) + [0] * len(self.negatives))
        self.n_samples = len(self.samples)
    
    def loadDataset(self, fpath):
        fpath = os.path.join(SENTEVAL_DATA_BASE, fpath)
        with io.open(fpath, 'r', encoding='latin-1') as f:
            return [line.split() for line in f.read().splitlines()]

    
class CRDataset(Dataset):
    def load_positives(self):
        self.positives = self.loadDataset('CR/custrev.pos')

    def load_negatives(self):
        self.negatives = self.loadDataset('CR/custrev.pos')

class MRDataset(Dataset):
    def load_positives(self):
        self.positives = self.loadDataset('MR/rt-polarity.pos')

    def load_negatives(self):
        self.negatives = self.loadDataset('MR/rt-polarity.neg')

class SUBJDataset(Dataset):
    def load_positives(self):
        self.positives = self.loadDataset('SUBJ/subj.objective')

    def load_negatives(self):
        self.negatives = self.loadDataset('SUBJ/subj.subjective')

class MPQADataset(Dataset):
    def load_positives(self):
        self.positives = self.loadDataset('MPQA/mpqa.pos')

    def load_negatives(self):
        self.negatives = self.loadDataset('MPQA/mpqa.neg')    