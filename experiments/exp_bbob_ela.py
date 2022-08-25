#Experiment with ELA and bbob classes
import pandas as pd
import os
from sklearn import manifold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import bbobbenchmarks as bbob
from doe2vec import doe_model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             multilabel_confusion_matrix)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelBinarizer


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


ela = pd.read_excel('featELA_BBOB_original.xlsx', index_col=0)
ela = ela.fillna(0)
#print(ela.Response1.values)

#for column in ela.columns[:]:
#    print(column)

encodings = []
fuction_groups = []
fuction_nrs = []
response = 1
for f in range(1, 25):
    for i in range(50):
        encodings.append(ela[f"Response{response}"].values)
        response += 1
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
        fuction_nrs.append(f)
        fuction_groups.append(class_label)


X = np.array(encodings)
y = np.array(fuction_groups).flatten()
fuction_nrs = np.array(fuction_nrs).flatten()
print(X.shape, y.shape)
print(np.unique(fuction_groups))
# y_dense = LabelBinarizer().fit_transform(y)

rf = RandomForestClassifier(n_estimators=100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf.fit(X_train, y_train)
resRf = rf.predict(X_test)

plot_confusion_matrix(
    y_test, resRf, np.unique(fuction_groups), title="Random Forest Confusion Matrix_ELA"
)


X = np.array(encodings)
y = np.array(fuction_nrs).flatten()
mds = manifold.MDS(
    n_components=2,
    random_state=0,
)
embedding = mds.fit_transform(X).T
# display a 2D plot of the bbob functions in the latent space

plt.figure(figsize=(12, 10))
plt.scatter(embedding[0], embedding[1], c=y, cmap=cm.jet)
plt.colorbar()
plt.xlabel("")
plt.ylabel("")

plt.savefig("latent_space_ela.png")
