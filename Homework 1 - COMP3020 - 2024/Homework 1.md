# Homework 1 Handout.

Contents:

- [Getting Started](#getting-started)
  - [Decision tree Implementation](#decision-tree-implementation)
  - [K-nn Implementation](#k_nn-implementation)
  - [Classification](#classification)

We will start this Homework by having you solve implementing Decision tree.

## Decision tree Implementation

Open the ```decision_tree.py``` file, this shoudld contain a skeleton of the decision
tree class that you need to implement. Here's the summary of the things you need to 
implement:

 - The `predict` function: traversing the root node, returning the majority class
 from a leaf.
 - The `_find_best_split` function: using the splits from the `split_generator` function to determine which feature and at what point to make the splits. For the
 first exercise, you only need to implement `ig` which stands for Information Gain.
 - The `_fit_tree` function: create the root node,  iterate through the dataset
 to find the left and right node for the tree recursively, till the depth limited is
 reached, or no more leaves can be split.
 - The `_calculate_entropy` function: implement the entropy function used for information gain.
 
 Once you implemented Information Gain, run a demo function to see how your method works:
 
`python decision_tree.py --demo ig`

This implementation also has a mode for Gini.

To run a few test case we will used to mark your implemetation score, use the `test.py` script.

`python test.py --model tree --mode ig `

## k-Nearest Neighbors Implementation.
Open the ```k_nearest_neighbors.py``` file, this shoudld contain a skeleton of the k-NN class that you need to implement. Here's the summary of the things you need to 
implement:

 - The `predict` function: Implement the prediction which returns the majority class for k-nearest element in the training set.
 - **Bonus**: Implement `predict` function in the `EpsilonballNearestNeighborClassifier` class.

 Once you implemented k-NN, run a demo function to see how your method works:
 
`python k_nearest_neighbors.py --demo knn`

This implementation also has a mode for Epsilon Ball.

To run a few test case we will used to mark your implemetation score, use the `test.py` script.

`python test.py --model nn --mode knn `

## Classification

Open the ```classification.py``` file, this should not be contain any code except for a description of the problem. Here's the summary of the things you need to 
implement:

 - Loading the preprocessing the data, this include:
   - Convert labels into integers.
   - Dealing with `NaN` values.
 - Run the classification task on both Decision tree and K-nn.
 - Displaying the results for both on the validation set.
 - Demonstrating and explaining the concept of underfitting and overfitting, using either the two models you implemented.

You will be judge based on the steps you implemented in the `classification.py ` and your analysis. 

If you struggle with this exercise, a tutorial is private which helps you with your first steps.

## And that's all folks!

Thanks for having made it here. Goodluck and Have fun.