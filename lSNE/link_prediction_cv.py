import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report

def link_prediction_cv(node_vector, edges, op, n_splits=5, random_state=1):
    # Setup the k-fold cross-validation configuration
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    if len(node_vector) == 2:
        B = node_vector[0]
        W = node_vector[1]
    else:
        B = node_vector[0]
        W = node_vector[0]

    # Prepare to collect scores
    f1_scores = []
    roc_scores = []

    # Split data using indices
    for train_index, test_index in kf.split(edges, edges[:, 2]):
        train_edges = edges[train_index]
        test_edges = edges[test_index]
        
        # Create training and testing feature sets
        X_train = op(B[train_edges[:, 0]], W[train_edges[:, 1]])
        Y_train = train_edges[:, 2]
        X_test = op(B[test_edges[:, 0]], W[test_edges[:, 1]])
        Y_test = test_edges[:, 2]
        
        # Initialize and fit the logistic regression model
        clf = LogisticRegression(class_weight='balanced', random_state=random_state)
        clf.fit(X_train, Y_train)
        
        # Predict and calculate scores
        test_preds = clf.predict(X_test)
        f1 = f1_score(Y_test, test_preds)
        roc = roc_auc_score(Y_test, clf.decision_function(X_test))
        
        # Collecting results
        f1_scores.append(f1)
        roc_scores.append(roc)
    
    # Calculate and print the average scores
    avg_f1 = np.mean(f1_scores)
    avg_roc = np.mean(roc_scores)
    print("Average F1 score: {:.3f}".format(avg_f1))
    print("Average ROC AUC score: {:.3f}".format(avg_roc))
    
    return avg_f1, avg_roc

# Example usage:
# node_vector = [B, W] # Your node embeddings
# edges = np.array(...) # Your edges array in the form of [node1, node2, label]
# op = lambda u, v: u * v  # An example operation on node embeddings
# f1, roc = link_prediction_cv(node_vector, edges, op)
