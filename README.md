# Decision Tree Coursework – COMP70050 (Introduction to Machine Learning)

This repository contains the implementation and evaluation of a Decision Tree Classifier for indoor room localisation using WiFi signal strengths. The work follows the COMP70050 coursework specification. 

The code trains the decision tree using the information gain splitting rule, choosing the feature and value that maximizes information gain under entropy rules. For this, we use a 10-fold cross-validation and report average performance over these 10 evaluations. We do this for both noisy and clean data and for pruned and full trees to see how pruning affects generalization and test performance. The results shows that for the noisy data, the model generalizes better with pruning. 

---

## Repository Overview

The repository contains the following files:
- *img* is the folder where the visualizations are exported to. Our four images are four the combinations of $(\text{prune}, \text{no prune}) \times (\text{clean data}, \text{noisy data})$.
- *wifi_db* is the folder containing the clean and noisy sensory wifi data, along with a label-column describing which room the signal is recorded from.
- *train.py* trains and evaluates the tree, outputing confusion matrix, accuracy, precision, recall, F1 and average max depth.
- *visualize.py* declares the function visualize_tree which is used in train.py to store images in the img folder. 

---


## Run code
```bash
python train.py --data "wifi_db/clean_dataset.txt" --visualize
python train.py --data "wifi_db/clean_dataset.txt" --prune --visualize
python train.py --data "wifi_db/noisy_dataset.txt" --visualize
python train.py --data "wifi_db/noisy_dataset.txt" --prune --visualize
