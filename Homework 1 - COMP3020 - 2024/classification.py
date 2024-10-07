
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from k_nearest_neighbors import KNNClassifier
from decision_tree import DecisionTree

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


if __name__ == "__main__":
    """
        For this exercise, you will be loading and training the models
        you implemented, that being K-nearest neighbors and Decision tree.
        
        You are given the Water Quality and Potability dataset, the main
        goal of which is to predict whether or not the drink sample is Potable
        or not.
        
        You are provided with two subsets:
             - Training dataset -  to train the model.
             - Validation datasest - to validate the model after training.
        
        To earn points for this exercise, you need to:
            1. Load the datasets and preprocess it.
            2. Train the model.
            3. Demonstrate overfiting and underfiting through the training and 
            validation results. This can be done by:
                - Show different results using different hyper-parameters.
                - Plot out the changes between training and validation accuracy
                when increasing certain hyper-parameters.
        
        If you find it difficult to get started, please follow the tutorial
        given in the ```classification_tutorial.ipynb``` notebook.
    """
    ### YOUR CODE HERE
    # Load in the dataset
    df_train = pd.read_csv('./dataset/train.csv')
    df_val = pd.read_csv('./dataset/val.csv')
    
    # Print the first five samples to get an idea of the dataset
    print("-----------------------------------------------------------------------------------------")   
    print("train dataset")
    print("-----------------------------------------------------------------------------------------")
    print(df_train.head())
    print("-----------------------------------------------------------------------------------------")
    print("validation dataset")
    print("-----------------------------------------------------------------------------------------")
    print(df_val.head())
    print("-----------------------------------------------------------------------------------------")

    # Convert the labels in the "TRN_TYPE" collumn into numbers
    le = preprocessing.LabelEncoder()
    le.fit(df_train["TRN_TYPE"])
    le.fit(df_val["TRN_TYPE"])

    # Check how many unique elements in the collumn.
    # The return list corresponds to the list of classes, 
    # and the index of each element represents that element's 
    # converted number.
    print(f"List of unique elements:{le.classes_}")

    # To convert from string labels to int labels, use le.transform
    df_train["TRN_TYPE"] = le.transform(df_train["TRN_TYPE"])
    df_val["TRN_TYPE"] = le.transform(df_val["TRN_TYPE"])

    # print out the first 5 elements:
    print("-----------------------------------------------------------------------------------------")   
    print("train dataset")
    print("-----------------------------------------------------------------------------------------")
    print(df_train.head())
    print("-----------------------------------------------------------------------------------------")
    print("validation dataset")
    print("-----------------------------------------------------------------------------------------")
    print(df_val.head())
    print("-----------------------------------------------------------------------------------------")

    # replace the Nan variables with "0"
    df_train = df_train.fillna(0)
    df_val = df_val.fillna(0)

    # The label will be the "TRN_TYPE" collumn, and the features will be the rest of the collumns.
    label_collumn = "TRN_TYPE"
    y_train = df_train[label_collumn]
    X_train = df_train.loc[:, df_train.columns != label_collumn]

    y_val = df_val[label_collumn]
    X_val = df_val.loc[:, df_val.columns != label_collumn]

    # convert the collumns to numpy arrays
    y_train = y_train.to_numpy()
    X_train = X_train.to_numpy()

    y_val = y_val.to_numpy()
    X_val = X_val.to_numpy()

    # train the model with decision tree
    train_accuracies_tree = []
    val_accuracies_tree = []
    hyper_param_tree = [1, 2, 3, 4, 5, 10, 15, 20, 50, 100]
    for C in hyper_param_tree:
        tree = DecisionTree(max_depth=C,mode="ig")
        tree.fit(X_train,y_train)

        y_pred_tree_train = tree.predict(X_train)
        train_accuracies_tree.append(accuracy_score(y_pred_tree_train,y_train))

        y_pred_tree_val = tree.predict(X_val)
        val_accuracies_tree.append(accuracy_score(y_pred_tree_val,y_val))

    # plt.plot(hyper_param_tree, train_accuracies_tree, label='Training Accuracy')
    # plt.plot(hyper_param_tree, val_accuracies_tree, label='Validation Accuracy')
    # plt.xlabel('Hyper-parameter C')
    # plt.ylabel('Accuracy')
    # # plt.xscale('log')
    # plt.legend()
    # plt.title('Training and Validation Accuracy on different max_depth in decision tree')
    # plt.show()


    # train the model with KNN
    train_accuracies_knn = []
    val_accuracies_knn = []
    hyper_param_knn = [2, 4, 6, 8, 10, 15, 20, 50]
    for C in hyper_param_knn:
        knn = KNNClassifier(k=C, ord=2)
        knn.fit(X_train,y_train)

        y_pred_knn_train = knn.predict(X_train)
        train_accuracies_knn.append(accuracy_score(y_pred_knn_train,y_train))

        y_pred_knn_val = knn.predict(X_val)
        val_accuracies_knn.append(accuracy_score(y_pred_knn_val,y_val))

    # plt.plot(hyper_param_knn, train_accuracies_knn, label='Training Accuracy')
    # plt.plot(hyper_param_knn, val_accuracies_knn, label='Validation Accuracy')
    # plt.xlabel('Hyper-parameter C')
    # plt.ylabel('Accuracy')
    # # plt.xscale('log')
    # plt.legend()
    # plt.title('Training and Validation Accuracy on different k in KNN')
    # plt.show()
    # knn = KNNClassifier(k=3, ord=2)
    # knn.fit(X=X_train, y=y_train)

    # y_pred_tree = tree.predict(X_val)
    # y_pred_knn = knn.predict(X_val)

    # print("-----------------------------------------------------------------------------------------")
    # print(f"Valiation accuracy on decision tree:{accuracy_score(y_pred_tree,y_val):.2%}")
    # print("-----------------------------------------------------------------------------------------")
    # print(f"Valiation accuracy on KNN:{accuracy_score(y_pred_knn,y_val):.2%}")

    fig, axs = plt.subplots(1, 2, figsize=(14,5))

    # Plot for Decision Tree
    axs[0].plot(hyper_param_tree, train_accuracies_tree, label='Training Accuracy')
    axs[0].plot(hyper_param_tree, val_accuracies_tree, label='Validation Accuracy')
    axs[0].set_xlabel('Hyper-parameter max_depth')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].set_title('Training and Validation Accuracy on different max_depth in Decision Tree')

    # Plot for KNN
    axs[1].plot(hyper_param_knn, train_accuracies_knn, label='Training Accuracy')
    axs[1].plot(hyper_param_knn, val_accuracies_knn, label='Validation Accuracy')
    axs[1].set_xlabel('Hyper-parameter k')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].set_title('Training and Validation Accuracy on different k in KNN')

    plt.tight_layout()
    plt.show()