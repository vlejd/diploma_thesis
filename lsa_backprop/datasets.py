import os, io
import numpy as np

SENTEVAL_DATA_BASE = '/home/vlejd/Documents/school/diplomka/SentEval/data/senteval_data'
class Dataset(object):
    def __init__(self, seed=1111):
        self.seed = seed
        self.positives = self.load_positives()
        self.negatives = self.load_negatives()
        self.samples = self.positives + self.negatives
        self.labels = np.array([1] * len(self.positives) + [0] * len(self.negatives))
        self.n_samples = len(self.samples)
    
    def loadFile(self, fpath):
        with io.open(fpath, 'r', encoding='latin-1') as f:
            return [line.split() for line in f.read().splitlines()]

    
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