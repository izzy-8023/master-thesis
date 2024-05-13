import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, roc_auc_score


def main(data_dir):
    # Step 1: Determine the sign of the test set
    file1_path = os.path.join(data_dir, 'results/test.txt')
    file2_path = os.path.join(data_dir, 'slashdot.txt')
    output_path = os.path.join(data_dir, 'results/matching_results.txt')

    with open(file1_path, 'r') as file:
        file1_lines = file.readlines()
    file1_data = [line.strip().split() for line in file1_lines]

    with open(file2_path, 'r') as file:
        file2_lines = file.readlines()
    file2_data = [line.strip().split('\t') for line in file2_lines]

    matching_lines = []
    for line2 in file2_data:
        for line1 in file1_data:
            if line2[:2] == line1:
                matching_lines.append('\t'.join(line2))

    with open(output_path, 'w') as file:
        for line in matching_lines:
            file.write(line + '\n')

    # Step 2: Add the sign of each edge to train.log
    df1 = pd.read_csv(output_path, header=None, sep='\t')
    df2_path = os.path.join(data_dir, 'results/test.log')
    df2 = pd.read_csv(df2_path, header=None, sep='\t')
    df2[64] = df1[2]
    combined_file_path = os.path.join(data_dir, 'results/combined_file.txt')
    df2.to_csv(combined_file_path, header=None, index=False, sep='\t')

    ##balanced dataset. 
   
    # Step 3: Linear regression
    data = pd.read_csv(combined_file_path, delim_whitespace=True, header=None)  
    label_counts = data.iloc[:, -1].value_counts()
    print(label_counts)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_log_reg = log_reg.predict(X_test)
    y_pred_prob_log_reg = log_reg.predict_proba(X_test)[:, 1]  
    print("Logistic Regression F1:", f1_score(y_test, y_pred_log_reg))
    print("Logistic Regression AUC:", roc_auc_score(y_test, y_pred_prob_log_reg))

   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sign prediction for edges using logistic regression and random forest.")
    parser.add_argument('data_dir', type=str, help="Directory containing the dataset.")
    args = parser.parse_args()

    main(args.data_dir)




