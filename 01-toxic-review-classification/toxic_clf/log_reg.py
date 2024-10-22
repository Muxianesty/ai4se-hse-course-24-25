import datasets
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, log_loss, f1_score
from tqdm import tqdm

KFOLD_SPLITS = 10

def classifier(dataset: datasets.Dataset):
    X = np.array(dataset["message"])
    y = np.array(dataset["is_toxic"])

    skf = StratifiedKFold(n_splits=KFOLD_SPLITS, random_state=42, shuffle=True)
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(X).toarray()
    y_pred = np.array([None] * x.shape[0])
    model = LogisticRegression(penalty="l1", solver="liblinear", C=1.35 )
    for train_index, test_index in tqdm(skf.split(x, y)):
        model.fit(X=x[train_index], y=y[train_index])
        y_pred[test_index] = model.predict(x[test_index])

    
    loss = log_loss(y, y_pred)
    print(f"Mean cross-entropy loss: {loss}")
    rounded_y_pred = y_pred.astype(int)
    rounded_y_pred[rounded_y_pred < 0.5] = 0
    rounded_y_pred[rounded_y_pred >= 0.5] = 1
    c_matrix = confusion_matrix(y_true=y, y_pred=rounded_y_pred, labels=[0, 1])
    print(f"True non-toxic comments recognized: {c_matrix[0, 0]}")
    print(f"Non-toxic comments treated as toxic: {c_matrix[0, 1]}")
    print(f"Toxic comments treated as non-toxic: {c_matrix[1, 0]}")
    print(f"True toxic comments recognized: {c_matrix[1, 1]}")
    f1 = f1_score(y_true=y, y_pred=rounded_y_pred, labels=[0, 1])
    print(f"Final F1-score: {f1}")
