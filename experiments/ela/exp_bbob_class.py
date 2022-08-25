import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

import bbobbenchmarks as bbob
from doe2vec import doe_model
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             multilabel_confusion_matrix)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer
from CEOELA_main import run_ELA


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
f1s = []
for dim in [2,5,10,15,20,30,40,50,100]:

    obj = doe_model(
        dim, 8, n=100000, latent_dim=24, use_mlflow=False, model_type="VAE", kl_weight=0.001
    )
    if not obj.load("../../models/"):
        obj.generateData()
        obj.compile()
        obj.fit(100)
        obj.save("../../models/")
    #obj.plot_label_clusters_bbob()
    sample = obj.sample * 10 - 5
    encodings = []
    fuction_groups = []
    evaluated_landscapes = []

   
    for i in range(100):
         for f in range(1, 25):
            fun, opt = bbob.instantiate(f, i)
            bbob_y = np.asarray(list(map(fun, sample)))
            array_x = (bbob_y.flatten() - np.min(bbob_y)) / (
                np.max(bbob_y) - np.min(bbob_y)
            )
            encoded = obj.encode([array_x])
            evaluated_landscapes.append(array_x)
            encodings.append(encoded[0])
            class_label = 0
            if f in [1, 2, 3, 4, 5]:
                class_label = "separable"
            elif f in [6, 7, 8, 9]:
                class_label = "low cond."
            elif f in [10, 11, 12, 13, 14]:
                class_label = "high cond."
            elif f in [15, 16, 17, 18, 19]:
                class_label = "multi modal gl."
            elif f in [20, 21, 22, 23, 24]:
                class_label = "multi modal"

            fuction_groups.append(class_label)

    np.save(f"dims/{dim}-landscapes.npy",evaluated_landscapes)
    np.save(f"dims/{dim}-sample.npy", sample)

    X = np.array(encodings)
    y = np.array(fuction_groups).flatten()

    #write DOE data for ELA to excel
    
    input_names = []
    input_names_2 = []
    output_names = []
    for d in range(dim):
        input_names.append(f"DV{d+1}")
        input_names_2.append(f"DV{d+1}")
    for o in range(len(X)):
        output_names.append(f"Response{o+1}")
    df_kpi = pd.DataFrame({'input':[],	'input_rename':[],'output':[],'output_rename':[]})
    input_names.extend(['']*(len(output_names)-len(input_names)))
    df_kpi['input'] = input_names
    df_kpi['output'] = output_names
    df_bounds = pd.DataFrame({'design variable':input_names_2,'lower':[-5]*len(input_names_2),'nominal':[0]*len(input_names_2),'upper':[5]*len(input_names_2)})
    doe_dict = {}
    for d in range(dim):
        doe_dict[f"DV{d+1}"] = sample[:,d]
    print(len(sample[:,0]))
    for o in range(len(evaluated_landscapes)):
        doe_dict[f"Response{o+1}"] = evaluated_landscapes[o]
    print(len(evaluated_landscapes[0]))
    df_doe = pd.DataFrame(doe_dict)
    with pd.ExcelWriter(f'ela-d{dim}.xlsx') as writer:
        df_kpi.to_excel(writer, sheet_name='KPI',index=False)
        df_bounds.to_excel(writer, sheet_name='Bounds',index=False)
        df_doe.to_excel(writer, sheet_name='DOE_1',index=False)

    run_ELA(f'ela-d{dim}.xlsx', f'd{dim}')

    rf = RandomForestClassifier(n_estimators=100)
    X_train = X[:-200]
    X_test = X[-200:]
    y_train = y[:-200]
    y_test = y[-200:]

    rf.fit(X_train, y_train)
    resRf = rf.predict(X_test)

    plot_confusion_matrix(
        y_test, resRf, np.unique(fuction_groups), title=f"Random Forest Confusion Matrix VAE d{dim}"
    )
    f1_macro_rf = f1_score(y_test, resRf, average='macro')
    f1s.append(f1_macro_rf)
    print(dim, f1_macro_rf)
    # plot_confusion_matrix(mul_dt, np.unique(fuction_groups))

print(f1_macro_rf)
np.save(f"f1macro.npy", f1_macro_rf)



