# FIRST: RENAME THIS FILE TO sentiment_utils.py 

#Harishraj Udaya Bhaskar, Aushee Khamesra




"""
Felix Muzny
CS 4/6120
Homework 4
Fall 2023

Utility functions for HW 4, to be imported into the corresponding notebook(s).

Add any functions to this file that you think will be useful to you in multiple notebooks.
"""
# fancy data structures
from collections import defaultdict, Counter
# for tokenizing and precision, recall, f_measure, and accuracy functions
import nltk
# for plotting
import matplotlib.pyplot as plt
# so that we can indicate a function in a type hint
from typing import Callable
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
nltk.download('punkt')

def generate_tuples_from_file(training_file_path: str) -> list:
    """
    Generates data from file formated like:

    tokenized text from file: [[word1, word2, ...], [word1, word2, ...], ...]
    labels: [0, 1, 0, 1, ...]
    
    Parameters:
        training_file_path - str path to file to read in
    Return:
        a list of lists of tokens and a list of int labels
    """
    # PROVIDED
    f = open(training_file_path, "r", encoding="utf8")
    X = []
    y = []
    for review in f:
        if len(review.strip()) == 0:
            continue
        dataInReview = review.strip().split("\t")
        if len(dataInReview) != 3:
            continue
        else:
            t = tuple(dataInReview)
            if (not t[2] == '0') and (not t[2] == '1'):
                print("WARNING")
                continue
            X.append(nltk.word_tokenize(t[1]))
            y.append(int(t[2]))
    f.close()  
    return X, y


"""
NOTE: for all of the following functions, we have prodived the function signature and docstring, *that we used*, as a guide.
You are welcome to implement these functions as they are, change their function signatures as needed, or not use them at all.
Make sure that you properly update any docstrings as needed.
"""

def get_prfa(dev_y: list, preds: list, verbose=False):
    """
    Calculate precision, recall, f1, and accuracy for a given set of predictions and labels.
    Args:
        dev_y: list of labels
        preds: list of predictions
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    precision = precision_score(dev_y, preds)
    recall = recall_score(dev_y, preds)
    f1 = f1_score(dev_y, preds)
    accuracy = accuracy_score(dev_y, preds)

    if verbose:
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Accuracy: {accuracy}")

    return precision, recall, f1, accuracy

def create_training_graph(metrics_fun: Callable, train_feats: list, dev_feats: list, Train_labels,True_labels:list, kind: str, savepath: str = None, verbose: bool = False) -> None:
    """
    Create a graph of the classifier's performance on the dev set as a function of the amount of training data.
    Args:
        metrics_fun: a function that takes in training data and dev data and returns a tuple of metrics
        train_feats: a list of training data in the format [(feats, label), ...]
        dev_feats: a list of dev data in the format [(feats, label), ...]
        kind: the kind of model being used (will go in the title)
        savepath: the path to save the graph to (if None, the graph will not be saved)
        verbose: whether to print the metrics
    """
    #TODO: implement this function
    training_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Lists to store performance metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []

    # Train and evaluate the classifier for different percentages of training data
    for percentage in training_percentages:
        # Select a subset of training data
        train_size = int(train_feats.shape[0] * percentage / 100)
        X_train_subset = train_feats[:train_size]
        y_train_subset = Train_labels[1][:train_size]

        # Create and train the Logistic Regression model
        if kind == 'Logistic':
            model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
            model.fit(X_train_subset, y_train_subset)

        y_pred = model.predict(dev_feats)
        p, r, f, a = metrics_fun(True_labels, y_pred)

        # Calculate and store performance metrics
        precision_scores.append(p)
        recall_scores.append(r)
        f1_scores.append(f)
        accuracy_scores.append(a)
        print(precision_scores)
        print(recall_scores)
        print(f1_scores)
        print(accuracy_scores)

    # Plot the results
    plt.figure(figsize=(10, 6))

    plt.plot(training_percentages, precision_scores, label='Precision')
    plt.plot(training_percentages, recall_scores, label='Recall')
    plt.plot(training_percentages, f1_scores, label='F1 Score')
    plt.plot(training_percentages, accuracy_scores, label='Accuracy')

    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Performance on Dev Set')
    plt.title(f'{kind} Performance vs. Training Data Size')
    plt.legend()
    plt.grid(True)

    if savepath:
        plt.savefig(savepath)
    plt.show()

def create_training_graph_lr(X_train,y_train,X_test,y_test):
    # Percentage of training data to use
    training_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Lists to store performance metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []

    # Train and evaluate the classifier for different percentages of training data
    for percentage in training_percentages:
        # Select a subset of training data
        train_size = int(X_train.shape[0] * percentage / 100)
        X_train_subset = X_train[:train_size]
        y_train_subset = y_train[:train_size]

        # Create and train the Logistic Regression model
        model = LogisticRegression(max_iter=1000,solver='lbfgs', multi_class='multinomial')
        model.fit(X_train_subset, y_train_subset)

        # Make predictions on the dev set
        y_pred = model.predict(X_test)
        p,r,f,a=get_prfa(y_test,y_pred)

        # Calculate and store performance metrics
        precision_scores.append(p)
        recall_scores.append(r)
        f1_scores.append(f)
        accuracy_scores.append(a)

    # Plot the results
    plt.figure(figsize=(10, 6))

    plt.plot(training_percentages, precision_scores, label='Precision')
    plt.plot(training_percentages, recall_scores, label='Recall')
    plt.plot(training_percentages, f1_scores, label='F1 Score')
    plt.plot(training_percentages, accuracy_scores, label='Accuracy')

    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Performance on Dev Set')
    plt.title('Classifier Performance vs. Training Data Size')
    plt.legend()
    plt.grid(True)
    plt.show()



def create_training_graph_nn(X_train,y_train,X_test,y_test,no_of_epochs=3):

    # Percentage of training data to use
    training_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Lists to store performance metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []

    # Train and evaluate the classifier for different percentages of training data
    for percentage in training_percentages:
        # Select a subset of training data
        train_size = int(X_train.shape[0] * percentage / 100)
        X_train_subset = X_train[:train_size]
        y_train_subset = np.asarray(y_train[:train_size])

        # Create and train the Neural Network model
        model = Sequential()
        model.add(Dense(50, input_dim=X_train_subset.shape[1], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

        model.fit(X_train_subset, y_train_subset, epochs=no_of_epochs, batch_size=32, validation_data=(X_test, np.asarray(y_test)))

        # Make predictions on the dev set
        predictions = model.predict(X_test)
        classifications = (predictions > 0.5).astype(int)
        p, r, f, a = precision_score(np.asarray(y_test), classifications), recall_score(np.asarray(y_test), classifications), f1_score(np.asarray(y_test), classifications), accuracy_score(np.asarray(y_test), classifications)

        # Calculate and store performance metrics
        precision_scores.append(p)
        recall_scores.append(r)
        f1_scores.append(f)
        accuracy_scores.append(a)

    # Plot the results
    plt.figure(figsize=(10, 6))

    plt.plot(training_percentages, precision_scores, label='Precision')
    plt.plot(training_percentages, recall_scores, label='Recall')
    plt.plot(training_percentages, f1_scores, label='F1 Score')
    plt.plot(training_percentages, accuracy_scores, label='Accuracy')

    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Performance on Dev Set')
    plt.title('Classifier Performance vs. Training Data Size')
    plt.legend()
    plt.grid(True)
    plt.show()
    return (f1_scores)




def featurize(vocab: list, data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> list:
    """
    Create vectorized BoW representations of the given data.
    Args:
        vocab: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: boolean for whether or not to print out progress
    Returns:
        a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    # using a Counter is essential to having this not take forever
    #TODO: implement this function
    labeled_data = []
    
    for i in range(len(vocab)):
        if binary:
            # Binary features
            features = {word: True for word in vocab[i]}
        else:
            # Multinomial features
            word_freq = Counter(vocab[i])
            total_words = sum(word_freq.values())
            
            # Check for empty documents
            if total_words > 0:
                features = {word: freq / total_words for word, freq in word_freq.items()}
            else:
                features = {}
        
        labeled_tuple = (features, vocab[i])
        labeled_data.append(labeled_tuple)
    
    return labeled_data
