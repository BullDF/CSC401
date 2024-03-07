#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2024 Frank Rudzicz, Gerald Penn

import argparse
import os
from scipy.stats import ttest_ind
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

# set the random state for reproducibility 
import numpy as np

np.random.seed(401)

clfs = [
    'SGDClassifier()',
    'GaussianNB()',
    'RandomForestClassifier(n_estimators=10, max_depth=5)',
    'MLPClassifier(alpha=0.05)',
    'AdaBoostClassifier()'
]


def accuracy(C):
    """ Compute accuracy given NumPy array confusion matrix C. Returns a floating point value. """
    total = np.sum(C)
    correct = 0

    for i in range(len(C)):
        correct += C[i, i]

    return correct / total if total != 0 else 0


def recall(C):
    """ Compute recall given NumPy array confusion matrix C. Returns a list of floating point values. """
    result = np.zeros(len(C))

    denoms = np.sum(C, axis=1)
    for j in range(len(C)):
        result[j] = C[j, j] / denoms[j] if denoms[j] != 0 else 0

    return result


def precision(C):
    """ Compute precision given NumPy array confusion matrix C. Returns a list of floating point values. """
    result = np.zeros(len(C))

    denoms = np.sum(C, axis=0)
    for i in range(len(C)):
        result[i] = C[i, i] / denoms[i] if denoms[i] != 0 else 0

    return result


def class31(output_dir, X_train, y_train, X_test, y_test):
    """ 
    This function performs experiment 3.1.

    Parameters:
    - output_dir: path of directory to write output to.
    - X_train: NumPy array, with the selected training features.
    - y_train: NumPy array, with the selected training classes.
    - X_test:  NumPy array, with the selected testing features.
    - y_test:  NumPy array, with the selected testing classes.

    Returns:
    - best_index: int, the index of the supposed best classifier.
    """

    print('Running task 3.1')

    max_acc = 0

    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        for i in range(len(clfs)):
            clf = eval(clfs[i])
            classifier_name = clf.__class__.__name__
            print(f'Training {classifier_name}')

            clf.fit(X_train, y_train)

            preds = clf.predict(X_test)
            conf_matrix = confusion_matrix(y_test, preds)
            acc = accuracy(conf_matrix)
            rec = recall(conf_matrix)
            prec = precision(conf_matrix)

            if acc > max_acc:
                max_acc = acc
                best_index = i

            # For each classifier, compute results and write the following output:
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')

    return best_index


def class32(output_dir, X_train, y_train, X_test, y_test, best_index):
    """
    This function performs experiment 3.2.

    Parameters:
    - output_dir: path of directory to write output to.
    - X_train: NumPy array, with the selected training features.
    - y_train: NumPy array, with the selected training classes.
    - X_test:  NumPy array, with the selected testing features.
    - y_test:  NumPy array, with the selected testing classes.
    - best_index:   int, the index of the supposed best classifier (from task 3.1).

    Returns:
       X_1k: NumPy array, just 1K rows of X_train.
       y_1k: NumPy array, just 1K rows of y_train.
    """

    print('Running task 3.2')

    sample_sizes = [1000, 5000, 10000, 15000, 20000]

    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        for num_train in sample_sizes:
            clf = eval(clfs[best_index])

            rng = np.random.default_rng(401)
            sample_index = rng.choice(len(X_train), size=num_train, replace=False)
            sample_X_train = X_train[sample_index]
            sample_y_train = y_train[sample_index]

            if num_train == 1000:
                X_1k, y_1k = sample_X_train, sample_y_train

            clf.fit(sample_X_train, sample_y_train)
            preds = clf.predict(X_test)
            conf_matrix = confusion_matrix(y_test, preds)
            acc = accuracy(conf_matrix)

            outf.write(f'{num_train}: {acc:.4f}\n')

    assert X_1k.shape == (1000, 173)
    assert y_1k.shape == (1000, )

    return X_1k, y_1k


def class33(output_dir, X_train, y_train, X_test, y_test, best_index, X_1k, y_1k):
    """
    This function performs experiment 3.3.

    Parameters:
    - output_dir: path of directory to write output to.
    - X_train: NumPy array, with the selected training features.
    - y_train: NumPy array, with the selected training classes.
    - X_test:  NumPy array, with the selected testing features.
    - y_test:  NumPy array, with the selected testing classes.
    - best_index: int, the index of the supposed best classifier (from task 3.1).
    - X_1k:    NumPy array, just 1K rows of X_train (from task 3.2).
    - y_1k:    NumPy array, just 1K rows of y_train (from task 3.2).
    """

    print('Running task 3.3')

    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.

        # for each number of features k_feat, write the p-values for
        # that number of features:

        for k_feat in [5, 50]:
            selector = SelectKBest(f_classif, k=k_feat)
            selector.fit(X_train, y_train)
            p_values = selector.pvalues_[selector.get_support(indices=True)]

            outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')

        selector_1k = SelectKBest(f_classif, k=5)
        X_1k_new = selector_1k.fit_transform(X_1k, y_1k)
        clf_1k = eval(clfs[best_index])
        clf_1k.fit(X_1k_new, y_1k)
        X_1k_test_new = selector_1k.transform(X_test)
        preds_1k = clf_1k.predict(X_1k_test_new)
        conf_matrix_1k = confusion_matrix(y_test, preds_1k)
        accuracy_1k = accuracy(conf_matrix_1k)

        selector = SelectKBest(f_classif, k=5)
        X_new = selector.fit_transform(X_train, y_train)
        clf = eval(clfs[best_index])
        clf.fit(X_new, y_train)
        X_test_new = selector.transform(X_test)
        preds = clf.predict(X_test_new)
        conf_matrix = confusion_matrix(y_test, preds)
        accuracy_full = accuracy(conf_matrix)

        assert clf_1k != clf

        feats_1k = selector_1k.get_support(indices=True)
        top_5 = selector.get_support(indices=True)

        feature_intersection = []
        for feat in feats_1k:
            if feat in top_5:
                feature_intersection.append(feat)

        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {top_5}\n')


def class34(output_dir, X_train, y_train, X_test, y_test, best_index):
    """
    This function performs experiment 3.4.

    Parameters:
    - output_dir: path of directory to write output to.
    - X_train: NumPy array, with the selected training features.
    - y_train: NumPy array, with the selected training classes.
    - X_test:  NumPy array, with the selected testing features.
    - y_test:  NumPy array, with the selected testing classes.
    - best_index:       int, the index of the supposed best classifier (from task 3.1).
    """

    print('Running task 3.4')

    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        #     outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        # outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')

        kf = KFold(n_splits=5, shuffle=True, random_state=401)

        accs = np.zeros((5, 5))

        for i in range(5):
            kfold_accuracies = np.zeros(5)
            for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
                clf = eval(clfs[i])
                print(f"{clf.__class__.__name__} fold {fold}")

                train_data = X_train[train_index]
                train_labels = y_train[train_index]
                val_data = X_train[val_index]
                val_labels = y_train[val_index]

                clf.fit(train_data, train_labels)
                preds = clf.predict(val_data)
                conf_matrix = confusion_matrix(val_labels, preds)
                kfold_accuracies[fold] = accuracy(conf_matrix)

            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
            accs[i, :] = kfold_accuracies

        p_values = []
        for i in range(5):
            if i != best_index:
                p_values.append(ttest_ind(accs[i], accs[best_index]).pvalue)

        outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="The input npz file from Task 2.", required=True)
    parser.add_argument(
        "-o", "--output-dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # TODO: load data and split into train and test.
    # TODO: complete each classification experiment, in sequence.
    feats = np.load(args.input)['arr_0']
    X, y = feats[:, :173], feats[:, 173]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=401, test_size=0.2)

    output_dir = args.output_dir
    best_index = class31(output_dir, X_train, y_train, X_test, y_test)
    X_1k, y_1k = class32(output_dir, X_train, y_train, X_test, y_test, best_index)
    class33(output_dir, X_train, y_train, X_test, y_test, best_index, X_1k, y_1k)
    class34(output_dir, X_train, y_train, X_test, y_test, best_index)
