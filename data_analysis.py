import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# import sys

# sys.path.append('./utils')
# # My files
# import utils

def get_data_and_target(e4_feature_list):
    """
        args:


        returns:
            X,Y -   data matrix and target vector
    """
    labels = list(e4_feature_list.keys())
    features = list(e4_feature_list[labels[0]].keys())

    num_labels = len(labels)
    num_features = len(features)
    num_experiments = len(e4_feature_list[labels[0]][features[0]])

    X = np.ndarray([num_experiments*num_labels, num_features])
    Y = np.ndarray([num_labels*num_experiments])

    for i, label in enumerate(labels):
        for j,feature in enumerate(features):
            X[i*num_experiments:(i+1)*num_experiments,j] = e4_feature_list[label][feature]
            Y[i*num_experiments:(i+1)*num_experiments] = np.ones([1,num_experiments])*i
        
    return X,Y


def remove_features(X, y, features, features_to_remove):
    """
    """
    indices = []
    features_new = copy.deepcopy(features)
    for feature in features_to_remove:
        i = features.index(feature)
        indices.append(i)
    X_new = np.delete(X,indices,1)
    for feature in features_to_remove:
        features_new.remove(feature)
    return X_new, features_new


def plot_loadings_heatmap(X, features, num_PCs = 0, title='loadings heatmap', fig_size=(8,8)):
    """
    """
    if num_PCs > 0:
        pca = PCA(num_PCs)
    else:
        pca = PCA()

    pca.fit(X)
    loadings = pca.components_.T
    columns = []


    for i in range(1,loadings.shape[1]+1):
        columns.append("PC" + str(i))
    loadings_plotable = pd.DataFrame(loadings, columns =columns , index = features)
    
    plt.figure(figsize = fig_size)
    sns.heatmap(loadings_plotable, yticklabels=True)
    plt.title(title)
    plt.show()


def get_pca_data(e4_feature_list, plot_heatmap = False, num_components = 0):
    # Generate input matrix for PCA:
    ## This will be a 3D matrix, where the first dim are observations, the second is features, and the third is labels.
    labels = list(e4_feature_list.keys())
    features = list(e4_feature_list[labels[0]].keys())

    num_labels = len(labels)
    num_features = len(features)
    num_experiments = len(e4_feature_list[labels[0]][features[0]])

    X = np.ndarray([num_experiments*num_labels, num_features])
    Y = np.ndarray([num_labels*num_experiments])

    for i, label in enumerate(labels):
        for j,feature in enumerate(features):
            X[i*num_experiments:(i+1)*num_experiments,j] = e4_feature_list[label][feature]
            Y[i*num_experiments:(i+1)*num_experiments] = np.ones([1,num_experiments])*i
    
    if num_components > 0:
        pca = PCA(num_components)
    else:
        pca = PCA()

    pca.fit(X)

    loadings = pca.components_.T
    if plot_heatmap:
        # Get columns, i.e. number of PC's
        columns = []
        for i in range(1,loadings.shape[1]+1):
            columns.append("PC" + str(i))
        loadings_plotable = pd.DataFrame(loadings, columns =columns , index = features)

        sns.heatmap(loadings_plotable, yticklabels=True)
    
    models = get_models(1,num_components)

    [best_model, best_degree, best_score, avg_scores, max_scores] = evaluate_models(models, X, Y)

    print(f"Best_deg: {best_degree}, best_score: {best_score}")

    plt.figure()
    plt.plot(range(1,num_components+1), avg_scores, label="Avg. scores")
    plt.plot(range(1,num_components+1), max_scores, label = "Max scores")
    plt.legend()
    plt.show()

#=========================
#   Model selection
#=========================

def get_models(min_PC, max_PC):
    models = {}
    for i in range(min_PC, max_PC + 1):
        steps = [('pca', PCA(i)), ('svc', SVC(kernel='rbf', decision_function_shape='ovo'))]
        pipeline = Pipeline(steps=steps)
        models[str(i)] = pipeline
    return models


def evaluate_models(models, data, target):
    best_model = ""
    best_score = 0
    best_degree = 0
    
    training_scores = []
    avg_scores = []
    min_scores = []
    max_scores = []
    std = []

    for degree, model in models.items():
        score = cross_val_score(model, data, target)

        fitted_model = model.fit(data,target)
        training_scores.append(fitted_model.score(data,target))
        avg_scores.append(np.mean(score))
        max_scores.append(np.max(score))
        min_scores.append(np.min(score))
        std.append(np.std(score))

        if np.mean(score) > best_score:
            best_score = np.mean(score)
            best_model = fitted_model
            best_degree = degree

    return [best_model, best_degree, best_score, np.array(training_scores), np.array(avg_scores), np.array(min_scores), np.array(max_scores), np.array(std)]


def plot_scores(avg_scores, max_scores, model_degrees):
    plt.figure()
    plt.plot(model_degrees, avg_scores)
    plt.plot(model_degrees, max_scores)
    plt.show()

