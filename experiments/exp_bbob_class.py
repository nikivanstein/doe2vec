import src.bbobbenchmarks as bbob
from src.doe2vec import doe_model
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_test,y_scores, classNames, title="confusion_matrix"):
    classes = len(classNames)
    cm = confusion_matrix(y_test, y_scores)
    print("**** Confusion Matrix ****")
    print(cm)
    print("**** Classification Report ****")
    print(classification_report(y_test, y_scores, target_names=classNames))
    con = np.zeros((classes,classes))
    for x in range(classes):
        for y in range(classes):
            con[x,y] = cm[x,y]/np.sum(cm[x,:])

    plt.figure(figsize=(20,20))
    sns.set(font_scale=3.0) # for label size
    df = sns.heatmap(con, annot=True,fmt='.2', cmap='Blues',xticklabels=classNames , yticklabels=classNames)
    plt.tight_layout()
    df.figure.savefig(title)

"""Classification experiment for BBOB
"""

obj = doe_model(5, 9, n=20000, latent_dim=16, use_mlflow=False)
if not obj.load("../models/"):
    obj.generateData()
    obj.compile()
    obj.fit(100)
    obj.save("../models/")
sample = obj.sample * 10 - 5

encodings = []
fuction_groups = []

for f in range(1,25):
    for i in range(50):
        fun, opt = bbob.instantiate(f,i)
        bbob_y =  np.asarray(list(map(fun, sample)))
        array_x = (bbob_y.flatten() - np.min(bbob_y)) / (
                    np.max(bbob_y) - np.min(bbob_y)
                )
        encoded = obj.encode([array_x])
        encodings.append(encoded[0])
        class_label = 0
        if (f in [1,2,3,4,5]):
            class_label = "separable"
        elif (f in [6,7,8,9]):
            class_label = "low cond."
        elif (f in [10,11,12,13,14]):
            class_label = "high cond."
        elif (f in [15,16,17,18,19]):
            class_label = "multi modal gl."
        elif (f in [20,21,22,23,24]):
            class_label = "multi modal"

        fuction_groups.append(class_label)

X = np.array(encodings)
y = np.array(fuction_groups).flatten()
print(X.shape, y.shape)
print(np.unique(fuction_groups))
#y_dense = LabelBinarizer().fit_transform(y)

dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt.fit(X_train, y_train)
res = dt.predict(X_test)

plot_confusion_matrix(y_test, res,np.unique(fuction_groups), title="Decision Tree Confusion Matrix" )
#mul_dt = multilabel_confusion_matrix(
#    y_test,
#    res,
#    labels=np.unique(fuction_groups))

rf.fit(X_train, y_train)
resRf = rf.predict(X_test)

plot_confusion_matrix(y_test, resRf, np.unique(fuction_groups), title="Random Forest Confusion Matrix" )

#plot_confusion_matrix(mul_dt, np.unique(fuction_groups))