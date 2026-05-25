# Weekly-Challenge-18-Random-Forest
#  Week 18: Random Forest from Scratch

##  Description
For Week 18, I decided to take on one of my biggest mathematical and programming challenges yet: building a **Random Forest Classifier** completely from scratch, using pure Python and NumPy, without relying on `scikit-learn`.

This project is deeply aligned with my Master's degree research on early detection algorithms for school dropout. By programming the core mechanics of ensemble learning myself, I gain a foundational understanding of how the algorithm makes its decisions.

##  How it works
The project is divided into two core classes:
1. **`DecisionTree`**: The base estimator. It recursively splits the data by calculating the **Information Gain** using the **Gini Impurity** metric:
   $$Gini = 1 - \sum (p_i)^2$$
   It finds the exact feature and threshold that best separates the classes (e.g., Dropout vs. Enrolled).
2. **`RandomForest`**: The ensemble manager. It utilizes **Bagging (Bootstrap Aggregating)**. It creates multiple subsets of the data with replacement, trains a separate `DecisionTree` on each subset, and then aggregates their predictions using a majority vote.

##  Complexity Analysis
* **Time Complexity:** $O(M \times N \log N \times D)$ for training, where $M$ is the number of trees, $N$ is the number of samples, and $D$ is the number of features.
* **Space Complexity:** $O(M \times K)$ where $K$ is the number of nodes in each tree.

## Dependencies
* Python 3.14.3
* NumPy
* Collections (Built-in)
