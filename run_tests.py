#!/usr/bin/env python3
import time
import os
import numpy as np
import pandas as pd
import openml
import json
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from gama.configuration.fasttextclassifier import FastTextClassifier
from gama import GamaClassifier
from gama.search_methods.asha import AsynchronousSuccessiveHalving
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve
from gama.postprocessing.ensemble import EnsemblePostProcessing
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler

def remove_non_string_cols(X):
  # Only keep the columns with string values
  print("Let's ditch the non-string columns broooo")
  return X[[col_name for col_name in X.columns if X[col_name].dtype == np.dtype('O')]]

def fasttext_run(X, y):
  #y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
  X = remove_non_string_cols(X)
  if isinstance(y, np.ndarray):
    y = pd.Series(LabelEncoder().fit_transform(y))
  else:
    y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

  clf = FastTextClassifier(minn=0, maxn=0, epoch=5, lr=0.1)
  clf.classes_ = y.unique()
  score = cross_val_score(clf, X_train, y_train, 
                          cv=2,
                          scoring=make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr", average="macro", labels=sorted(y.unique())),
                          n_jobs=2)
  print("CV AUC scores:", score)
  avg_score = sum(score) / len(score)
  print("average CV AUC score:", avg_score)
  return avg_score

def gama_run(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
  if min(y_train.value_counts()) < 10:
    ros = RandomOverSampler(sampling_strategy="minority")
    X_train, y_train = ros.fit_resample(X_train, y_train)
  automl = GamaClassifier(max_total_time=600,
    store="logs",
    max_eval_time=240,
    post_processing=EnsemblePostProcessing(),
    scoring="roc_auc_ovr"
  )
  print("Starting to fit.")
  automl.fit(X_train, y_train)
  #label_predictions = automl.predict(X_test)
  probability_predictions = automl.predict_proba(X_test)
  if probability_predictions.shape[1] == 2: 
    # if binary class
    probability_predictions = probability_predictions[:,1]
  gama_score = roc_auc_score(y_test, probability_predictions, average="macro", multi_class="ovr")
  print('ROC AUC:', gama_score)
  return gama_score

def log_error(e):
  with open("script_error.txt", "a+") as out:
    out.write(e)


def main(ids):
  scores = []
  for ds_name, d_id in ids.items():
      
    dataset_scores = {
      "data_id": d_id,
      "data_name": ds_name 
    }

    try:
      dataset = openml.datasets.get_dataset(d_id)
      X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
      )
      if d_id == 42076:
        y = X["state"]
        del X["state"]
      # Cap to 100.000 instances
      if len(X) > 100000:
        X, _, y, _ = train_test_split(X, y, stratify=y, train_size=100000, random_state=0)
    except Exception as e:
      msg = f"{d_id} openml failed: {e}\n"
      print(msg)
      log_error(msg)

#    try:
#      dataset_scores["fasttext"] = fasttext_run(X, y)
#    except Exception as e:
#      msg = f"{d_id} fasttext failed: {e}\n"
#      print(msg)
#      log_error(msg)

    try:
      dataset_scores["gama"] = gama_run(X, y)
    except Exception as e:
      msg = f"{d_id} GAMA failed: {e}\n"
      print(msg)
      log_error(msg)

    scores.append(dataset_scores)

  try:
    if not os.path.isdir("roc_results"):
      os.mkdir("roc_results")
    with open(f"roc_results/results_{int(time.time())}.json", "w+") as f:
      json.dump(scores, f)
  except Exception as e:
    print(e)

if __name__ == "__main__":
  # Getting the data set

  ids = {
    "beerreviews": 42078,
    "road_safety": 42803,
    "traffic_violations": 42132,
    "drug_directory": 43044,
    "kickstarter": 42076,
    "openpayments": 42738,
  }

  for _ in range(5):
    main(ids)

