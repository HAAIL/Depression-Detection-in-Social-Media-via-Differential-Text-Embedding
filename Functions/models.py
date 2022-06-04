
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import KFold
import numpy as np
import copy as cp
import matplotlib.pyplot as plt

import seaborn as sns
from typing import Tuple
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
# -----------------------------------------------------------------
# Create a function to train the models and report the perofmance
# -----------------------------------------------------------------


def cross_val_predict(model, kfold: KFold, X: np.array, y: np.array) -> Tuple[np.array, np.array, np.array]:
    model_ = cp.deepcopy(model)

    no_classes = len(np.unique(y))

    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes])

    for train_ndx, test_ndx in kfold.split(X):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]
        actual_classes = np.append(actual_classes, test_y)

        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))
        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    return actual_classes, predicted_classes, predicted_proba


def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, sorted_labels: list):

    matrix = confusion_matrix(actual_classes, predicted_classes)
    print(matrix)
    plt.figure(figsize=(12.8, 6))
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted');
    plt.ylabel('Actual');
    plt.title('Confusion Matrix')
    plt.savefig("cm")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
from numpy import interp
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, \
    precision_recall_curve
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC



from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.pipeline import Pipeline

ROS_pipeline = make_pipeline(RandomOverSampler(random_state=777),lr)
original_pipeline = Pipeline([('classifier', lr)])
SMOTE_pipeline = make_pipeline(SMOTE(random_state=777),lr)

def lr_cv(splits, X, Y, s):
    kfold = KFold(n_splits=splits, shuffle=True, random_state=777)
    accuracy = []
    precision = []
    recall = []
    f1 = []

    accuracyim = []
    precisionim = []
    recallim = []
    f1im = []
    auim = []
    model = LogisticRegression(penalty='l1', max_iter=10000000, solver='liblinear')
    for train, test in kfold.split(X, Y):
        # lr_fit = pipeline.fit(X[train], Y[train])
        # prediction = lr_fit.predict(X[test])
        smoter = SMOTE(random_state=42)

        X_train_fold, y_train_fold = X[train], Y[train]

        # Get the validation data
        X_val_fold, y_val_fold = X[test], Y[test]
        model_objim = model.fit(X_train_fold, y_train_fold)

        # Upsample only the data in the training section
        X_train_fold_upsample, y_train_fold_upsample = smoter.fit_resample(X_train_fold,
                                                                           y_train_fold)
        # Fit the model on the upsampled training data
        model_obj = model.fit(X_train_fold_upsample, y_train_fold_upsample)
        # Score the model on the (non-upsampled) validation data

        a = accuracy_score(y_val_fold, model_obj.predict(X_val_fold))
        accuracy.append(a)
        r = recall_score(y_val_fold, model_obj.predict(X_val_fold))
        recall.append(r)
        p = precision_score(y_val_fold, model_obj.predict(X_val_fold))
        precision.append(p)
        f = f1_score(y_val_fold, model_obj.predict(X_val_fold))
        f1.append(f)

        au = roc_auc_score(y_val_fold, model_objim.predict(X_val_fold))
        auim.append(au)
        a1 = accuracy_score(y_val_fold, model_objim.predict(X_val_fold))
        accuracyim.append(a1)
        r1 = recall_score(y_val_fold, model_objim.predict(X_val_fold))
        recallim.append(r1)
        p1 = precision_score(y_val_fold, model_objim.predict(X_val_fold))
        precisionim.append(p1)
        f2 = f1_score(y_val_fold, model_objim.predict(X_val_fold))
        f1im.append(f2)

    actual_classes, predicted_classes, _ = cross_val_predict(model_obj, kfold, X, Y)
    plot_confusion_matrix(actual_classes, predicted_classes, ["1", "0"])
    print("accuracy: %.3f%% (+/- %.2f%%)" % (np.mean(accuracy)*100, np.std(accuracy)))
    print("precision: %.3f%% (+/- %.2f%%)" % (np.mean(precision)*100, np.std(precision)))
    print("recall: %.3f%% (+/- %.2f%%)" % (np.mean(recall)*100, np.std(recall)))
    print("f1 score: %.3f%% (+/- %.2f%%)" % (np.mean(f1)*100, np.std(f1)))

    print("imbalance")
    actual_classes, predicted_classes, _ = cross_val_predict(model_objim, kfold, X, Y)
    plot_confusion_matrix(actual_classes, predicted_classes, ["1", "0"])
    print("accuracy: %.3f%% (+/- %.2f%%)" % (np.mean(accuracyim) * 100, np.std(accuracyim)))
    print("precision: %.3f%% (+/- %.2f%%)" % (np.mean(precisionim) * 100, np.std(precisionim)))
    print("recall: %.3f%% (+/- %.2f%%)" % (np.mean(recallim) * 100, np.std(recallim)))
    print("f1 score: %.3f%% (+/- %.2f%%)" % (np.mean(f1im) * 100, np.std(f1im)))
    print("f1 score: %.3f%% (+/- %.2f%%)" % (np.mean(auim) * 100, np.std(auim)))
    draw_cv_roc_curve(model_obj, kfold, X, y, title='Cross Validated ROC Of ' + s)
    draw_cv_pr_curve(model_obj, kfold, X, y, title='Cross Validated PR Curve Of ' + s)

    # predict probabilities





def run_exps(X, y, s):
    '''
    Lightweight script to test many models
    :param X_train: training split
    :param y_train: training target vector
    :param X_test: test split
    :param y_test: test target vector
    :return: DataFrame of predictions
    '''

    results = []
    names = []
    cms = []
    dfs = []
    models = [
        ('LogReg', LogisticRegression(penalty='l1', max_iter=10000000, solver='liblinear')),
    ]

    for name, model in models:
        print(s)
        print(name)
        kfold = KFold(n_splits=5, random_state=42, shuffle=True)

        mean_score = model_selection.cross_val_score(model, X, y, scoring="roc_auc", cv=kfold).mean()
        mean_acc = model_selection.cross_val_score(model, X, y, cv=kfold).mean()
        mean_recall = model_selection.cross_val_score(model, X, y, scoring="recall", cv=kfold).mean()
        mean_pre = model_selection.cross_val_score(model, X, y, scoring="precision", cv=kfold).mean()
        mean_f1 = model_selection.cross_val_score(model, X, y, scoring="f1", cv=kfold).mean()

        actual_classes, predicted_classes, _ = cross_val_predict(model, kfold, X, y)
        plot_confusion_matrix(actual_classes, predicted_classes, ["1", "0"])

        print("Mean accuracy: " + str(mean_acc))
        print("Mean roc_auc: " + str(mean_score))
        print("Mean recall: " + str(mean_recall))
        print("Mean precision: " + str(mean_pre))
        print("Mean f1: " + str(mean_f1))



