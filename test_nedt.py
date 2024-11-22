"""
Example for NEDT classifier: classify some random generated data using NEDT classifier.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from nedt import NEDTClassifier


def run_main():
    # generate some test data
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_classes=2,
        random_state=42,
        class_sep=0.5,
        weights=[0.5],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=123
    )

    # classify the data using NEDT
    clf = NEDTClassifier(
        criterion="entropy", max_depth=7, game_based=1, k_quantile_zero=0.2
    )
    clf.fit(X_train, y_train)
    clf_prediction, clf_predition_prob = clf.predict(X_test)

    print(f"auc for prediction: \t{roc_auc_score(y_test, clf_predition_prob)}")
    print(f"f1 for prediction: \t{f1_score(y_test, clf_prediction)}")
    print(f"accuracy for prediction:\t{accuracy_score(y_test, clf_prediction)}")
    print(f"log_loss for prediction: \t{log_loss(y_test, clf_prediction)}")


if __name__ == "__main__":
    run_main()
