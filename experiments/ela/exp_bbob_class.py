import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import autosklearn.classification

import bbobbenchmarks as bbob
from doe2vec import doe_model
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
f1s = []
f1s_elas = []
calc_ela = False
all_dims = [2,5,10,20,40]
latent_dim = 8
for dim in all_dims:

    obj = doe_model(
        dim, 32, n=250000, latent_dim=latent_dim, use_mlflow=False, model_type="VAE", kl_weight=0.001
    )
    
    if not obj.loadModel("../../models/"):
        if not obj.loadData("../../models/"):
            obj.generateData()
            obj.saveData("../../models/")
        tracker = EmissionsTracker(project_name=f"doe2vec-d{dim}", output_dir="../../models/")
        tracker.start()
        obj.compile()
        obj.fit(100, verbose=0)
        tracker.stop()
        obj.saveModel("../../models/")
    #obj.plot_label_clusters_bbob()
    sample = obj.sample * 10 - 5
    encodings = []
    fuction_groups = []
    evaluated_landscapes = []

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
            encoded = obj.encode([array_x])
            evaluated_landscapes.append(array_x)
            encodings.append(encoded[0])
            class_label = 0
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

    np.save(f"dims/{dim}-landscapes.npy",evaluated_landscapes)
    np.save(f"dims/{dim}-sample.npy", sample)

    X = np.array(encodings)
    y_1 = np.array(multim_label).flatten()
    y_2 = np.array(global_label).flatten()
    y_3 = np.array(funnel_label).flatten()

    #write DOE data for ELA to excel
    
    test_size = 20*25
    if calc_ela:
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
        #print(len(sample[:,0]))
        for o in range(len(evaluated_landscapes)):
            doe_dict[f"Response{o+1}"] = evaluated_landscapes[o]
        #print(len(evaluated_landscapes[0]))
        df_doe = pd.DataFrame(doe_dict)
        with pd.ExcelWriter(f'ela-d{dim}.xlsx') as writer:
            df_kpi.to_excel(writer, sheet_name='KPI',index=False)
            df_bounds.to_excel(writer, sheet_name='Bounds',index=False)
            df_doe.to_excel(writer, sheet_name='DOE_1',index=False)

        #if dim > 40:
        #check file exist(later)
        run_ELA(f'ela-d{dim}.xlsx', f'd{dim}')


        ela = pd.read_excel(f'CEOELA_results/d{dim}/featELA_d{dim}_original.xlsx', index_col=0)
        ela = ela.fillna(0)
        ela_encodings = []
        response = 1
        for i in range(120):
            for f in range(1, 25):
                ela_encodings.append(ela[f"Response{response}"].values)
                response+=1
        ela_X = np.array(ela_encodings)

    
    X_train = X[:-test_size]
    X_test = X[-test_size:]

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=240,
        per_run_time_limit=30,
        n_jobs=1,
        memory_limit=None
    )
    automl.fit(X_train, y_1[:-test_size], dataset_name='y1 doe2vec')
    resRf = automl.predict(X_test)
    f1_macro_rf = f1_score(y_1[-test_size:], resRf, average='macro')
    f1s.append(f1_macro_rf)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=240,
        per_run_time_limit=30,
        n_jobs=1,
        memory_limit=None
    )
    automl.fit(X_train, y_2[:-test_size], dataset_name='y2 doe2vec')
    resRf = automl.predict(X_test)
    f1_macro_rf = f1_score(y_2[-test_size:], resRf, average='macro')
    f1s.append(f1_macro_rf)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=240,
        per_run_time_limit=30,
        n_jobs=1,
        memory_limit=None
    )
    automl.fit(X_train, y_3[:-test_size], dataset_name='y3 doe2vec')
    resRf = automl.predict(X_test)
    f1_macro_rf = f1_score(y_3[-test_size:], resRf, average='macro')
    f1s.append(f1_macro_rf)

    #plot_confusion_matrix(
    #    y_test, resRf, np.unique(fuction_groups), title=f"Random Forest Confusion Matrix VAE d{dim}"
    #)
    
    print(dim, f1s)

    if calc_ela:
        #ELA model
        X_ela_train = ela_X[:-test_size]
        X_ela_test = ela_X[-test_size:]

        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=120,
            per_run_time_limit=30,
        )
        automl.fit(X_ela_train, y_1[:-test_size], dataset_name='y1 ela')
        resRf = automl.predict(X_ela_test)
        f1_macro_rf_ela = f1_score(y_1[-test_size:], resRf, average='macro')
        f1s_elas.append(f1_macro_rf_ela)

        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=120,
            per_run_time_limit=30,
        )
        automl.fit(X_ela_train, y_2[:-test_size], dataset_name='y2 ela')
        resRf = automl.predict(X_ela_test)
        f1_macro_rf_ela = f1_score(y_2[-test_size:], resRf, average='macro')
        f1s_elas.append(f1_macro_rf_ela)

        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=120,
            per_run_time_limit=30,
        )
        automl.fit(X_ela_train, y_3[:-test_size], dataset_name='y3 ela')
        resRf = automl.predict(X_ela_test)
        f1_macro_rf_ela = f1_score(y_3[-test_size:], resRf, average='macro')
        f1s_elas.append(f1_macro_rf_ela)

        print(dim, f1s_elas)

i = 0
for d in all_dims:
    print(d, f1s[i])
    print(d, f1s[i+1])
    print(d, f1s[i+2])
    i+=1

if calc_ela:
    print(f1s_elas)
    np.save("f1_ela.npy",f1s_elas)
np.save(f"f1_VAE-{latent_dim}.npy", f1s)




