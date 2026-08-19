from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .constants import FEATURE_COMBINATIONS
from .evaluation import evaluate, fit_evaluate, group_split
from .features import feature_matrix
from .io import parse_ids, resolve_columns


def count_matrix(counters, vocabulary):
    lookup = {value: index for index, value in enumerate(vocabulary)}
    rows, columns, values = [], [], []
    for row, counter in enumerate(counters):
        for key, value in counter.items():
            if key in lookup:
                rows.append(row)
                columns.append(lookup[key])
                values.append(float(value))
    return sp.csr_matrix((values, (rows, columns)), shape=(len(counters), len(vocabulary)), dtype=np.float32)


def semantic_baseline_matrices(trajectory, groups, columns, train):
    frame = pd.read_csv(trajectory)
    resolved = resolve_columns(frame, columns, ("group", "poi_categories", "tile_category"))
    lookup = {int(group): index for index, group in enumerate(groups)}
    poi_counts = [Counter() for _ in groups]
    tile_counts = [Counter() for _ in groups]
    for group, poi_categories, tile_category in frame[[resolved["group"], resolved["poi_categories"], resolved["tile_category"]]].itertuples(index=False, name=None):
        index = lookup[int(group)]
        poi_counts[index].update(parse_ids(poi_categories))
        tile_counts[index].update((int(tile_category),))
    poi_vocabulary = sorted({value for counter in poi_counts for value in counter})
    tile_vocabulary = sorted({value for counter in tile_counts for value in counter})
    poi = count_matrix(poi_counts, poi_vocabulary)
    tile = count_matrix(tile_counts, tile_vocabulary)
    transformer = TfidfTransformer(norm="l2")
    transformer.fit(poi[train])
    one_hot = tile.copy()
    one_hot.data[:] = 1.0
    return {
        "POI-category frequency": normalize(poi, norm="l1", axis=1),
        "POI-category TF-IDF": transformer.transform(poi),
        "tile-category one-hot": one_hot,
    }


def train_word2vec_matrix(trajectory, groups, columns, train, settings, seed=5):
    from gensim.models import Word2Vec

    frame = pd.read_csv(trajectory)
    resolved = resolve_columns(frame, columns, ("group", "poi_categories"))
    group_lookup = {int(group): index for index, group in enumerate(groups)}
    sequences = [[] for _ in groups]
    for group, value in frame[[resolved["group"], resolved["poi_categories"]]].itertuples(index=False, name=None):
        sequences[group_lookup[int(group)]].extend(str(item) for item in parse_ids(value))
    training_sequences = [sequences[index] for index in train if sequences[index]]
    model = Word2Vec(sentences=training_sequences, workers=1, seed=seed, **settings)
    matrix = np.full((len(groups), settings["vector_size"]), -1, dtype=np.float32)
    for index, sequence in enumerate(sequences):
        vectors = [model.wv[token] for token in sequence if token in model.wv]
        if vectors:
            matrix[index] = np.mean(vectors, axis=0)
    return matrix


def run_semantic_baselines(trajectory, features, columns, parameters, output, word2vec=None, word2vec_settings=None, seed=5, test_size=0.1, jobs=8):
    train, test = group_split(features["gr"], features["classes"], test_size, seed)
    matrices = semantic_baseline_matrices(trajectory, features["gr"], columns, train)
    matrices["DistMult-T"] = feature_matrix(features, "T")
    matrices["DistMult-T+PC"] = feature_matrix(features, "T+PC")
    if word2vec:
        matrices["Word2Vec"] = np.load(word2vec)
    elif word2vec_settings:
        matrices["Word2Vec"] = train_word2vec_matrix(trajectory, features["gr"], columns, train, word2vec_settings, seed)
    rows = []
    for name, matrix in matrices.items():
        _, _, metrics = fit_evaluate(matrix, features["classes"], train, test, parameters, seed, jobs)
        rows.append({"method": name, "dimension": matrix.shape[1], **{key: value for key, value in metrics.items() if key not in {"class_accuracy", "confusion_matrix"}}})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "semantic_baselines.csv", index=False, encoding="utf-8-sig")
    return rows


def run_classifier_comparison(features, parameters, output, combination=None, seed=5, test_size=0.1, jobs=8):
    labels = features["classes"]
    train, test = group_split(features["gr"], labels, test_size, seed)
    tuning = feature_matrix(features, "T+P+TC+PC")[train]
    cross_validation = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    svm_search = GridSearchCV(
        Pipeline((("scale", StandardScaler()), ("svc", SVC(kernel="rbf", gamma="scale")))),
        {"svc__C": [0.1, 1, 10, 100]},
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=jobs,
    )
    svm_search.fit(tuning, labels[train])
    combinations = (combination,) if combination else FEATURE_COMBINATIONS
    rows = []
    for current in combinations:
        matrix = feature_matrix(features, current)
        classifiers = {
            "Random Forest": RandomForestClassifier(max_depth=40, n_estimators=100, min_samples_leaf=10, random_state=5, n_jobs=jobs),
            "RBF-SVM": Pipeline((("scale", StandardScaler()), ("svc", SVC(kernel="rbf", gamma="scale", C=svm_search.best_params_["svc__C"])))),
        }
        _, _, xgb_metrics = fit_evaluate(matrix, labels, train, test, parameters, seed, jobs)
        rows.append({"classifier": "XGBoost", "feature_combination": current, **{key: value for key, value in xgb_metrics.items() if key not in {"class_accuracy", "confusion_matrix"}}})
        for name, classifier in classifiers.items():
            classifier.fit(matrix[train], labels[train])
            metrics = evaluate(labels[test], classifier.predict(matrix[test]))
            rows.append({"classifier": name, "feature_combination": current, **{key: value for key, value in metrics.items() if key not in {"class_accuracy", "confusion_matrix"}}})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "classifier_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(svm_search.cv_results_).to_csv(output / "svm_parameter_selection.csv", index=False, encoding="utf-8-sig")
    return rows
