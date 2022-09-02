import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import autosklearn.classification

import bbobbenchmarks as bbob
from doe2vec import doe_model
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             multilabel_confusion_matrix)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer
from CEOELA_main import run_ELA
from codecarbon import EmissionsTracker


def plot_confusion_matrix(y_test, y_scores, classNames, title="confusion_matrix"):
    classes = len(classNames)
    cm = confusion_matrix(y_test, y_scores)
    print("**** Confusion Matrix ****")
    print(cm)
    print("**** Classification Report ****")
    print(classification_report(y_test, y_scores, target_names=classNames))
    con = np.zeros((classes, classes))
    for x in range(classes):
        for y in range(classes):
            con[x, y] = cm[x, y] / np.sum(cm[x, :])

    plt.figure(figsize=(20, 20))
    sns.set(font_scale=3.0)  # for label size
    df = sns.heatmap(
        con,
        annot=True,
        fmt=".2",
        cmap="Blues",
        xticklabels=classNames,
        yticklabels=classNames,
    )
    plt.tight_layout()
    df.figure.savefig(title)


"""Classification experiment for BBOB
"""
f1_results = {}
calc_ela = False
all_dims = [2,5,10,20,40]
latent_dim = 24
for model_type in ["AE", "VAE"]:
    for latent_dim in [16,24,32]:
        f1_results[model_type+str(latent_dim)] = {}
        for dim in all_dims:
            obj = doe_model(
                dim, 9, n=250000, latent_dim=latent_dim, use_mlflow=False, model_type=model_type, kl_weight=0.001
            )
            
            sample = obj.sample * 10 - 5
            encodings = []
            fuction_groups = []
            evaluated_landscapes = []
            X = []
            multim_label = []
            global_label = []
            funnel_label = []
            for i in range(120):
                for f in range(1, 25):
                    fun, opt = bbob.instantiate(f, i)
                    bbob_y = np.asarray(list(map(fun, sample)))
                    array_x = (bbob_y.flatten() - np.min(bbob_y)) / (
                        np.max(bbob_y) - np.min(bbob_y)
                    )
                    X.append(array_x)
                    if f in [1,2]:
                        multim_label.append("none")
                        global_label.append("none")
                        funnel_label.append("yes")
                    elif f in [3,4]:
                        multim_label.append("high")
                        global_label.append("strong")
                        funnel_label.append("yes")
                    elif f in [8,9]:
                        multim_label.append("low")
                        global_label.append("none")
                        funnel_label.append("yes")
                    elif f in [5,6,7,10,11,12,13,14]:
                        multim_label.append("none")
                        global_label.append("none")
                        funnel_label.append("yes")
                    elif f in [15,19]:
                        multim_label.append("high")
                        global_label.append("strong")
                        funnel_label.append("yes")
                    elif f in [16]:
                        multim_label.append("high")
                        global_label.append("medium")
                        funnel_label.append("none")
                    elif f in [17,18]:
                        multim_label.append("high")
                        global_label.append("medium")
                        funnel_label.append("yes")
                    elif f in [20]:
                        multim_label.append("medium")
                        global_label.append("deceptive")
                        funnel_label.append("yes")
                    elif f in [21]:
                        multim_label.append("medium")
                        global_label.append("none")
                        funnel_label.append("none")
                    elif f in [22]:
                        multim_label.append("low")
                        global_label.append("none")
                        funnel_label.append("none")
                    elif f in [23]:
                        multim_label.append("high")
                        global_label.append("none")
                        funnel_label.append("none")
                    elif f in [24]:
                        multim_label.append("high")
                        global_label.append("weak")
                        funnel_label.append("yes")
            obj.setData(np.array(X))
            obj.compile()
            obj.fit(100, verbose=0)
            encodings = obj.encode(X)
                    
            X = np.array(encodings)
            y_1 = np.array(multim_label).flatten()
            y_2 = np.array(global_label).flatten()
            y_3 = np.array(funnel_label).flatten()

            test_size = 20*25
            
            X_train = X[:-test_size]
            X_test = X[-test_size:]

            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(X_train, y_1[:-test_size])

            resRf = rf.predict(X_test)
            f1_macro = f1_score(y_1[-test_size:], resRf, average='macro')
            f1_results[model_type+str(latent_dim)][f"d{dim} multimodal"] = f1_macro

            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(X_train, y_2[:-test_size])
            resRf = rf.predict(X_test)
            f1_macro = f1_score(y_2[-test_size:], resRf, average='macro')
            f1_results[model_type+str(latent_dim)][f"d{dim} global"] = f1_macro

            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(X_train, y_3[:-test_size])
            resRf = rf.predict(X_test)
            f1_macro = f1_score(y_3[-test_size:], resRf, average='macro')
            f1_results[model_type+str(latent_dim)][f"d{dim} funnel"] = f1_macro

            print(f1_results)


with open('f1_results.json', 'w') as fp:
    json.dump(f1_results, fp)




