# Decision Tree Coursework – COMP70050 (Introduction to Machine Learning)

This repository contains the implementation and evaluation of a Decision Tree Classifier for indoor room localisation using WiFi signal strengths.  
The work follows the COMP70050 coursework specification. The 

---

## Repository Overview

The repository contains the following files:
- *img* is the folder where the visualizations are exported to. Our four images are four the combinations of $(\text{prune}, \texxt{no prune}) \times (\text{clean data}, \text{noisy data})$.

---

## 🧠 Overview

The project builds and evaluates decision trees on both **clean** and **noisy** datasets (`wifi_db/clean_dataset.txt`, `wifi_db/noisy_dataset.txt`), using **10-fold cross-validation**.  
It supports post-training **reduced-error pruning** and includes optional tree visualization for the clean dataset.

**Main file:** `main.py`  
**Core functionality:**  
- Decision tree learning with continuous features  
- 10-fold cross-validation  
- Accuracy, precision, recall, and F1-score per class  
- Reduced-error pruning   
- Tree visualization

---

## Run code
python intro_to_ml_1.py --data "wifi_db/clean_dataset.txt"

python intro_to_ml_1.py --data "wifi_db/clean_dataset.txt" --prune

python intro_to_ml_1.py --data "wifi_db/noisy_dataset.txt"

python intro_to_ml_1.py --data "wifi_db/noisy_dataset.txt" --prune

python intro_to_ml_1.py --data "wifi_db/clean_dataset.txt" --visualize

