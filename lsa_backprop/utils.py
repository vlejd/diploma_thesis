import datasets
import seaborn as sns
sns.set()

def subtract_baseline(df):
    for d in datasets.ALL_DATASETS + datasets.TREC_DATASETS:
        if d.name() in df.columns:
            df[d.name()] -= d.bias()

def tabular(text, caption=''):
    top = r"""
\begin{table}[H]
\begin{center}

"""
    bot = r"""
\caption["""+caption+r"""]{"""+caption+r"""}
\label{tab:}
\end{center}
\end{table}

"""
    return top + text + bot


def multireplace(txt, lis):
    for q,w in lis:
        txt = txt.replace(q,w)
    return txt

cm = sns.light_palette("green", as_cmap=True)
def color_positives(val):
    color = 'red' if val > 0 else 'black'
    return 'color: %s' % color


TREC = ['TRECDataset-ABBR', 'TRECDataset-DESC', 'TRECDataset-ENTY', 'TRECDataset-HUM', 'TRECDataset-LOC', 'TRECDataset-NUM']
NOTREC = ['CRDataset', 'MPQADataset', 'MRDataset', 'SUBJDataset']

