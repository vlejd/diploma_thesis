import datasets

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

