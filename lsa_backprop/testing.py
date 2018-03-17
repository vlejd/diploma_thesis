import numpy as np
from tqdm import tnrange
#from collections import defaultdict

def test_simple_model(model, dataset, iters=1, tag=None, results=None, dump=None):
    precisions = []
    for it in tnrange(iters):
        model.fit(dataset.train_samples(), dataset.train_labels())
        train_p = model.score(dataset.train_samples(), dataset.train_labels())
        test_p = model.score(dataset.test_samples(), dataset.test_labels())
        precisions.append((train_p, test_p))

    if dump is not None:
        dump[dataset.name()][('simple', tag)] = {
            'model': model,
            'results': precisions
        }

    train, test = list(zip(*precisions))
    print(dataset.name())
    if tag is not None and results is not None:
        results[dataset.name()][('simple', tag, 'train')] = np.mean(train)
        results[dataset.name()][('simple', tag, 'test')] = np.mean(test)


    print('Train precision', np.min(train), np.mean(train), np.max(train))
    print('Test precision', np.min(test), np.mean(test), np.max(test))


