import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

METRIC = "f1_macro"
TEXT_COL = "text"
NUM_COLS = ["n_chars", "n_words"]


def fit_predict(train, test, cfg):
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True, max_features=50000, dtype=np.float32)
    xtr = vec.fit_transform(train[TEXT_COL].fill_null("").to_list())
    xte = vec.transform(test[TEXT_COL].fill_null("").to_list())
    ntr = np.log1p(train.select(NUM_COLS).to_numpy().astype(np.float32))
    nte = np.log1p(test.select(NUM_COLS).to_numpy().astype(np.float32))
    mu, sd = ntr.mean(0), ntr.std(0) + 1e-6
    xtr = hstack([xtr, (ntr - mu) / sd]).tocsr()
    xte = hstack([xte, (nte - mu) / sd]).tocsr()
    clf = LogisticRegression(C=4.0, max_iter=2000, random_state=cfg["seed"])
    clf.fit(xtr, train[cfg["target_col"]].to_numpy())
    return clf.predict(xte)
