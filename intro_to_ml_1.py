import numpy as np
import matplotlib.pyplot as plt
import argparse
from viz_tree_intro_ml import visualize_tree

# Entropy Function
# Takes in a tuple of label counts "counts" and returns the entropy
def H(counts):
  p = counts / counts.sum()
  p = p[p > 0] # Filters out 0 probabilities
  entropy = -sum(p*np.log2(p))
  return entropy


# Information gain function using running counts
# Takes a column of data "X_col" and the labels associated with each points "label"
# Returns the best average entropy "best_H_after" along with the associated best treshold for splitting the column "best_thresh"
def information_gain(X_col, label):

    # Sort X_col and label
    sorted_indices = np.argsort(X_col)
    X_col = X_col[sorted_indices]
    label = label[sorted_indices]

    best_H_after, best_thresh = -np.inf, None

    # Define starting positions for running counts
    left_counts = np.zeros(len(np.unique(label)))
    right_counts = np.unique(label, return_counts=True)[1]

    # Mapping from label to index to account for arrays such as (1, 3, 4)
    label_to_index = {label: index for index, label in enumerate(np.unique(label))}

    for i in range(len(X_col) - 1):
      current_label = int(label[i])
      left_counts[label_to_index[current_label]] += 1
      right_counts[label_to_index[current_label]] -= 1

      # If the following number is the same as the current one then there is no new threshold
      if X_col[i] == X_col[i+1]:
        continue

      next_threshold = (X_col[i] + X_col[i+1]) / 2

      H_left = H(left_counts)
      H_right = H(right_counts)

      # Weighted average entropy after split based on next_threshold
      H_after = (left_counts.sum()/len(label)) * H_left + (right_counts.sum()/len(label)) * H_right

      # Information gain
      gain = - H_after

      if gain > best_H_after:
          best_H_after = gain
          best_thresh = next_threshold

    return best_H_after, best_thresh


# Function to find the best split point for the dataset as a whole
# Takes in the dataset to be split "data"
# Returns the best feture "best_feature" and the best threshold "best_thresh" to split on
def find_split(data):
  features, label = data[:,:-1], data[:,-1]
  best_H, best_thresh = -np.inf, None
  best_feature = None

  for i in range(len(features[0,:])):
    h, thresh = information_gain(features[:,i], label)
    if h > best_H:
      best_H = h
      best_thresh = thresh
      best_feature = i

  return best_feature, best_thresh



# Function for building the decision tree
# Takes in the dataset as a whole "data" and a starting depth for the top node "depth"
def decision_tree_learning(data, depth):

  # Is Leaf Check
  if (len(np.unique(data[:,-1])) == 1):
    return (data[0,-1] , depth)

  # Find the best splitting point
  best_feature, best_thresh = find_split(data)

  left_data = data[data[:,best_feature] <= best_thresh]
  right_data = data[data[:,best_feature] > best_thresh]

  # Recursively split the dataset
  left_data, depth_left = decision_tree_learning(left_data, depth+1)
  right_data, depth_right = decision_tree_learning(right_data, depth+1)

  return (best_feature, best_thresh, left_data, right_data), max(depth_left, depth_right)


# Evaluation

def traverse_tree(tree,row):
  # take the tree structure, ignore depth
  if isinstance(tree, tuple) and len(tree) == 2:
    node = tree[0]
  else:
    node = tree

  # until we hit a leaf
  while isinstance(node, tuple) and len(node) == 4:
    feature, thresh, left, right = node

    # pick a node
    if row[feature] <= thresh:
      node = left
    else:
      node = right

  return node

# using the specs variable names
def evaluate(test_db, trained_tree):
  features = test_db[:, :-1]
  labels = test_db[:,-1]

  #compare each tree prediction to the labels for the test set
  predictions = np.array([traverse_tree(trained_tree, row) for row in features])
  accuracy = np.mean(predictions == labels)

  return accuracy, labels, predictions

def confusion_matrix(labels, predictions):
  confusion_matrix = np.zeros((4,4))

  # for each label,prediciton pair. increment the location the location in the matrix
  for i in range(len(labels)):
    confusion_matrix[int(labels[i])-1,int(predictions[i])-1] += 1

  return confusion_matrix


def cross_validate(dataset, toPrune):
  # split dataset by indexes into 10 folds
  idx = np.arange(len(dataset))
  folds = np.array_split(idx, 10)
  n_folds = len(folds)

  # store results for each fold
  fold_accuracies = []
  total_confusion_matrix = np.zeros((4,4))

  # loop through each fold
  for i in range(n_folds):
    test_idx = folds[i]
    test_db = dataset[test_idx]

    if ( not toPrune):
      # combine all other folds into the training set
      train_idx = []

      for j in range(n_folds):
        if j != i:
          train_idx.extend(folds[j])
      train_idx = np.array(train_idx)

      # split the dataset into trianing and test sets
      test_db = dataset[test_idx]
      train_db = dataset[train_idx]

      # train the tree on training data
      tree = decision_tree_learning(train_db, 0)[0]

    # Code for pruning
    if toPrune:
      lowest_error = (np.inf, i, 0)
      for j in range(10):
        if j == i: # Don't use the current test data as a validation set
          continue

        pruning_validation_data = dataset[folds[j]]
        prune_idx = []

        for k in range(10):
            if k != i and k != j:
                prune_idx.extend(folds[k])
        pruning_data = dataset[prune_idx]

        tree = decision_tree_learning(pruning_data, 0)[0]

        # Prune the tree based on different folds being the validation set
        pruned_tree = prune_tree(tree, pruning_data, pruning_validation_data)
        error = 1 - evaluate(pruning_validation_data, pruned_tree)[0]
        if lowest_error[0] > error:
          lowest_error = (error, j, pruned_tree)

      # Use the best pruned tree
      best_pruned_tree = lowest_error[2]
      tree = best_pruned_tree

    # evaluate the trained tree on the test data
    accuracy, labels, predictions = evaluate(test_db, tree)

    # store accuracy and update total confusion matrix
    fold_accuracies.append(accuracy)
    total_confusion_matrix += confusion_matrix(labels, predictions)

  mean_accuracy = np.mean(fold_accuracies)

  return total_confusion_matrix, mean_accuracy, tree

def precision_recall_f1(confusion_matrix):
  precision = []
  recall = []
  f1 = []

  # number of rooms in our scenario
  n_labels = len(confusion_matrix)

  # calculate the true-positive, false-postive and false-negative for each label
  for i in range(n_labels):
    # the diagonal is the set of tp
    tp  = confusion_matrix[i][i]

    # add the column for that label and remove the tp
    fn = 0
    for j in range(n_labels):
      fn += confusion_matrix[i][j]
    fn -= tp

    # add the row for that label and remove the tp
    fp = 0
    for j in range(n_labels):
      fp += confusion_matrix[j][i]
    fp -= tp

    # dont allow divide by 0 errors for each equation
    if tp + fp > 0:
      precision.append(tp/(tp+fp))
    else:
      precision.append(0)

    if tp + fn > 0:
      recall.append(tp/(tp+fn))
    else:
        recall.append(0)

    if precision[i] + recall[i] >0:
      f1.append(2 * precision[i] * recall[i] / (precision[i] + recall[i]))
    else:
      f1.append(0)

  return precision, recall, f1


# Funtion for pruning an existing decision tree
# Takes in the tree "tree", some data to assign majority labels "train_data", and a validation set for testing the pruned tree "validation_data"
# Returns the pruned tree
def prune_tree(tree, train_data, validation_data):
    # Return if the tree is just one leaf
    if isinstance(tree, (float, np.float64)):
        return tree

    feature, thresh, left, right = tree

    # If its not a leaf recursively find the left and right nodes
    left = prune_tree(left, train_data[train_data[:, feature] <= thresh], validation_data)
    right = prune_tree(right, train_data[train_data[:, feature] > thresh], validation_data)

    # Upon returning values for left and right check if both are leaves
    if isinstance(left, np.float64) and isinstance(right, np.float64):
        # Evaluate current error
        current_accuracy, _, _ = evaluate(validation_data, (feature, thresh, left, right))
        current_error = 1 - current_accuracy

        # Based on the training data find the majority label to assign to the new pruned node
        train_subset = train_data[(train_data[:, feature] <= thresh) | (train_data[:, feature] > thresh)]
        labels = train_subset[:, -1]

        # Doesn't prune if no test data
        if len(labels) == 0:
           return (feature, thresh, left, right)
        else:
            # Logic to assign majority_label accounting for edge cases of unique label arrays such as [1, 3, 4] returning incorrect majority_labels
            unique_labels = np.unique(labels)
            label_to_index = {label: index for index, label in enumerate(unique_labels)}
            mapped_labels = np.array([label_to_index[label] for label in labels])
            majority_label = unique_labels[np.bincount(mapped_labels).argmax()]
        pruned_tree = np.float64(majority_label)

        # Evaluate pruned error
        pruned_accuracy, _, _ = evaluate(validation_data, pruned_tree)
        pruned_error = 1 - pruned_accuracy

        # Only take the pruned solution if the pruned error is less than the current error
        if pruned_error <= current_error:
            return pruned_tree

    # Return the node
    return (feature, thresh, left, right)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to dataset txt file')
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--prune', action='store_true', help='Enable reduced-error pruning (nested CV)')
    parser.add_argument('--visualize', action='store_true',
                    help='Render a visualization of a tree trained on the FULL dataset')
    args = parser.parse_args()

    dataset = np.loadtxt(args.data)

    np.random.RandomState(args.seed).shuffle(dataset)

    confusion_matrix, accuracy, tree = cross_validate(dataset, args.prune)
    precision, recall, f1 = precision_recall_f1(confusion_matrix)

    

    print("Seed Used: ", args.seed, "\n")

    print("Pruned : ", args.prune, "\n")

    print("Max Depth: ", tree[1])

    print("Confusion Marix: \n", confusion_matrix,"\nAccuracy: \n", accuracy,"\nPrecision By Label: \n",precision,"\nRecall By Label: \n", recall,"\nF1 By Label: \n", f1, "\n")

    if (args.visualize):
      X, y = dataset[:,:-2], dataset[:,-1] 
      feature_names = [f"AP{i}" for i in range(X.shape[1])]
      visualize_tree(
            tree,
            feature_names=feature_names,
            title=f"Decision Tree trained on {args.data}",
            data_path=args.data,   # autosave next to dataset folder
            show_fig=False
        )
       
    
if __name__ == '__main__':
    # Allow importing this file as a module without running
    try:
        main()
    except SystemExit:
        # argparse will call sys.exit; ensure clean exit in notebooks
        pass
