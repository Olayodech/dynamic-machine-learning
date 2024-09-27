import pandas as pd;
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.calibration import LabelEncoder

from sklearn.cluster import KMeans;
from sklearn.preprocessing import StandardScaler;
from scipy.cluster.hierarchy import dendrogram, linkage;
from sklearn.metrics import silhouette_score, homogeneity_score

import warnings;
# warnings.filterwarnings('ignore');

data = pd.read_csv(r'../../../Customers.csv');

# print(data.head());
# print(data.describe());
# print(data.shape);
# print(data.info)
# print(data.isnull().sum())

def eda(data):
    sns.heatmap(data.isnull(), cbar=True, cmap="Blues_r")
    sns.barplot(data.duplicated())
    
    fig, axes  = plt.subplots(2,2, figsize=(1,1));
    sns.boxplot(data["Age"], ax=axes[0,0])
    sns.boxplot(data["CustomerID"], ax=axes[0,1])
    sns.boxplot(data["Spending_Score"], ax=axes[1,0])
    sns.boxplot(data["Annual_Income_(k$)"], ax=axes[1,1])
    
    plt.tight_layout();
    # plt.show()
    
def data_clean(data):
    data = data.dropna();
    data = data.drop(['CustomerID'], axis=1, inplace=True);
    cleaned_data = data;
    return cleaned_data;

def data_preprocess(data: pd.DataFrame):
    encoder = LabelEncoder();
    
    for c in data.columns:
        print(data[c].dtypes, data[c].name)
        if data[c].dtypes=="object":
            data[c] = encoder.fit_transform(data[c])
        else:
            data[c] = data[c]
    
    print(data.head());
    
def define_cluster_bin(data: pd.DataFrame):
    wcss = [];
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1,1), wcss, color="red", marker="0");
    plt.show();
    
    
def visualization(data: pd.DataFrame):
    plt.scatter(data["Age"], data["Annual_Income_(k$)"], c=data["Spending_Score"], s=50, cmap="viridis");
    plt.xlabel("Age");
    plt.ylabel("Annual Income (k$)");
    plt.colorbar();
    
    # should count number of clusters
    silhouette_score(data, KMeans.labels_); 
    plt.show();
    
    
    
    


eda(data);
cleaned_data = data_clean(data);
data_preprocess(data);
define_cluster_bin(cleaned_data)
visualization(cleaned_data);