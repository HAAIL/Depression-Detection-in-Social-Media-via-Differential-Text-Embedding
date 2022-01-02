
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

# -----------------------------------------------------------------
# Create a function to train the models and report the perofmance
# -----------------------------------------------------------------



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
        ('LogReg', LogisticRegression(penalty='none', max_iter=100000, dual=False)),
    ]

    for name, model in models:
        print(s)
        print(name)

        mean_score = model_selection.cross_val_score(model, X, y, scoring="roc_auc", cv=10).mean()
        mean_acc = model_selection.cross_val_score(model, X, y, cv=10).mean()
        mean_recall = model_selection.cross_val_score(model, X, y, scoring="recall", cv=10).mean()
        mean_pre = model_selection.cross_val_score(model, X, y, scoring="precision", cv=10).mean()
        mean_f1 = model_selection.cross_val_score(model, X, y, scoring="f1", cv=10).mean()

        print("Mean accuracy: " + str(mean_acc))
        print("Mean roc_auc: " + str(mean_score))
        print("Mean recall: " + str(mean_recall))
        print("Mean precision: " + str(mean_pre))
        print("Mean f1: " + str(mean_f1))


