import argparse
import pandas as pd
import os
import subprocess
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import shutil
import networkx as nx
import random
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def run_edge2vec(data_dir):
    total_graph_path = os.path.join(data_dir, 'results/total.graph')
    command = f"python {os.path.join(data_dir, 'edge2vec.py')} -i {total_graph_path} -m results -n 1000 -s 500"
    result = subprocess.run(command, shell=True, check=True)
    


def label_and_process_files(data_dir):
    
    
    df_train = pd.read_csv(os.path.join(data_dir, 'results/train.txt'), sep=' ', header=None, dtype=int)
    df_test = pd.read_csv(os.path.join(data_dir, 'results/test.txt'), sep=' ', header=None, dtype=int)
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    df_combined.to_csv(os.path.join(data_dir, 'results/combined_train.txt'), sep=' ', index=False, header=False)

    # Generate Negative Samples
    positive_edges = set(tuple(x) for x in df_combined[[0, 1]].values)

    # 获取节点的最大编号
    max_node_id = max(df_combined[0].max(), df_combined[1].max())

    # 创建正样本的边集
    positive_edges = {tuple(x) for x in df_combined[[0, 1]].values}

    # 创建所有可能的边集合
    all_possible_edges = {(i, j) for i in range(1, max_node_id + 1) for j in range(i + 1, max_node_id + 1)}

    # 计算每个节点的度
    degrees = {node: 0 for node in range(1, max_node_id + 1)}
    for node in pd.concat([df_combined[0], df_combined[1]]):
        if node != 0:  # 跳过编号为0的节点
            degrees[node] += 1

    # 创建一个概率分布，使得边缘的选择概率与其端点的度成反比，这可以一定程度上模拟同分布的假设
    # 即节点度越高，成为负样本的概率越低
    edge_selection_probability = {}
    for edge in all_possible_edges - positive_edges:
        if degrees[edge[0]] == 0 or degrees[edge[1]] == 0:
            edge_selection_probability[edge] = 0.0001  # 设置一个较小的非零值
        else:
            edge_selection_probability[edge] = 1 / (degrees[edge[0]] * degrees[edge[1]])

    # 提取概率分布的值，构建一维数组
    probabilities = np.array(list(edge_selection_probability.values()))

    # 根据概率分布选择负样本
    negative_sample_edges = np.random.choice(
        len(probabilities), 
        size=len(positive_edges), 
        replace=False, 
        p=probabilities / probabilities.sum()
    )

    # 从索引中获取负样本的边
    negative_sample_edges = [list(edge_selection_probability.keys())[i] for i in negative_sample_edges]

    # 将负样本转换为DataFrame
    df_negative_samples = pd.DataFrame(negative_sample_edges, columns=['Node1', 'Node2'])

    # 打印负样本的前几条记录
    print(df_negative_samples.head())

#############################

    #转换为DataFrame
    df_negative_samples = pd.DataFrame(list(negative_sample_edges), columns=['Node1', 'Node2'])
    df_negative_samples.to_csv(os.path.join(data_dir, 'results/negative_samples.txt'), sep=' ', index=False, header=False)

    # Combine again two files
    df_total = pd.concat([df_combined, df_negative_samples], ignore_index=True)
    df_total.to_csv(os.path.join(data_dir, 'results/total.txt'), sep=' ', index=False, header=False, float_format='%.0f')
    shutil.copy(os.path.join(data_dir, 'results/total.txt'), os.path.join(data_dir, 'results/total.graph'))

    # Embedding again; Run edge2vec
    run_edge2vec(data_dir)

    # Label embedded files
    df_train = pd.read_csv(os.path.join(data_dir, 'results/train.txt'), sep=' ', header=None)
    df_negative_samples = pd.read_csv(os.path.join(data_dir, 'results/negative_samples.txt'), sep=' ', header=None)
    set_of_tuples_df_negative_samples = set(map(tuple, df_negative_samples.values))
    df_train['label'] = df_train.apply(lambda row: 0 if (row[0], row[1]) in set_of_tuples_df_negative_samples else 1, axis=1)
    df_train.to_csv(os.path.join(data_dir, 'results/train_with_labels.txt'), sep=' ', index=False, header=False)
    label_counts = df_train['label'].value_counts()
    print(label_counts)

    train_with_labels_file_path = os.path.join(data_dir, 'results/train_with_labels.txt')
    train_log_file_path = os.path.join(data_dir, 'results/train.log')
    train_log_with_labels_file_path = os.path.join(data_dir, 'results/train_with_labels.log')

    train_txt_df = pd.read_csv(train_with_labels_file_path, header=None, sep=' ')
    train_log_df = pd.read_csv(train_log_file_path, header=None, sep=' ')
    train_log_df['label'] = train_txt_df.iloc[:, -1]
    train_log_df.to_csv(train_log_with_labels_file_path, header=False, index=False, sep=' ')


    # Train logistic regression model
    df_train_with_labels_log = pd.read_csv(os.path.join(data_dir, 'results/train_with_labels.log'), header=None, sep=' ')
    X = df_train_with_labels_log.iloc[:, :-1]
    y = df_train_with_labels_log.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    print(f'accuracy: {accuracy}')
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f'AUC: {auc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and analyze data.")
    parser.add_argument('data_dir', type=str, help="Directory containing the data files.")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"The directory {args.data_dir} does not exist.")
    else:
        label_and_process_files(args.data_dir)