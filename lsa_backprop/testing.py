import numpy as np
from tqdm import tnrange
#from collections import defaultdict

def test_simple_model(model, dataset, iters=1, tag=None, results=None, dump=None):
    precisions = []
    for it in tnrange(iters):
        model.fit(dataset.train_samples(), dataset.train_labels())
        train_p = model.score(dataset.train_samples(), dataset.train_labels())
        valid_p = model.score(dataset.valid_samples(), dataset.valid_labels())
        test_p = model.score(dataset.test_samples(), dataset.test_labels())
        precisions.append((train_p, valid_p, test_p))

    if dump is not None:
        dump[dataset.name()][('simple', tag)] = {
            'model': model,
            'results': precisions
        }

    train, valid, test = list(zip(*precisions))
    print(dataset.name())
    if tag is not None and results is not None:
        results[dataset.name()][('simple', tag, 'train')] = np.mean(train)
        results[dataset.name()][('simple', tag, 'valid')] = np.mean(valid)
        results[dataset.name()][('simple', tag, 'test')] = np.mean(test)


    print('Train precision', np.min(train), np.mean(train), np.max(train))
    print('Valid precision', np.min(valid), np.mean(valid), np.max(valid))
    print('Test precision', np.min(test), np.mean(test), np.max(test))
