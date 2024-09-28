import pandas as pd;
import numpy as np;
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler;
from sklearn.preprocessing import LabelEncoder;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import mean_squared_error, r2_score;
from sklearn.cluster import KMeans;
import plotly.express as px;

data = pd.read_csv(r'../../../../Country-data.csv');

def univariate_eda(data: pd.DataFrame):
    print(data.isnull().count())
    print(data.head(2));
    column_present = data.columns[1:];
    # plt.figure(2,2)
    # sns.heatmap(data);
    
    feature_len = len(column_present);
    fig, axes = plt.subplots(3,3, figsize=(20,15));
    position = axes.flatten();
    for i, column in enumerate(column_present):
        myPlot = sns.histplot(data[column], kde=True, ax=position[i]);
        plt.title(f"Distribution of {column}");
        myPlot.set_xlabel(column, title="", );
        
def bivariate(data: pd.DataFrame):
    sns.scatterplot(x = data["life_expec"], y=data["income"]);
    sns.regplot(x = data["life_expec"], y=data["income"]);
    # return column_present;
    
def top_five_country_child_mortality(data: pd.DataFrame):
    sorted_data = data.sort_values(by=["child_mort"], ascending=False);
    return sorted_data.head(5)

def least_five_country_child_mortality(data: pd.DataFrame):
    sorted_data = data.sort_values(by=["child_mort"], ascending=True).head(5);
    return sorted_data.head(5)

def feature_engineering(data: pd.DataFrame):
    new_data = data;
    new_data["Health"] = (
        (data["life_expec"] / data["life_expec"].mean()) 
        + (data["health"] / data["health"].mean()) 
        - (data["total_fer"] / data["total_fer"].mean())
        - (data["child_mort"] / data["child_mort"].mean())
    );
    
    new_data["Trade"] = (data["imports"]/ data["imports"].mean());
    new_data["finance"] = (data["income"]/ data["income"].mean());
    # print(new_data.head());
    
    scaler = StandardScaler();
    # new_data["Health"] = scaler.fit_transform(new_data["Health"]);
    # new_data["Trade"] = scaler.fit_transform(new_data["Trade"]);
    # new_data["Finace"] = scaler.fit_transform(new_data["Finance"]);
    print(new_data.head());
    return new_data;
    
def modelling(data):
    # clustering
    wcss = [];
    for i in range(1, 11):
        model = KMeans(i, init="k-means++", random_state=42, max_iter=1000);
        model.fit(data); #no label in unsupervised
        
        # within-cluster sum of squares (wcss)
        wcss.append(model.inertia_);
        # view the elbow method for error -- where the elbow falls is our optimal cluster number;
        plt.plot(range(1,11), wcss, marker="o", linestyle="__");
        
        cluster = model.cluster_centers_();
        labels = model.labels();
        
        data["class"] = model.labels_();  #target variable
        # print(data.head());
        plt.figure(figsize=(12,6));
        plt.subplot(1,2,1)
        sns.barplot(x="class", y="income", data=data);
        sns.barplot(x="class", y="child_mort", data=data);
        
        # sns.barplot(x="class", y="child_mort", data=data);
        
       # geographical map plot with choloropleth from plotly;
        if data["class"] == "1":
            data["class"][data["class"]==1] = "High Priority require foreign aid"
        elif data["class"] == "0":
            data["class"][data["class"]==0] = "Medium Priority require financial aid"
        else:
            data["class"][data["class"]==2] = "Low Priority require no aid";
            
        fig = px.choropleth(data[["country", "class"]], locationmode="country names", locations="country", color=data["class"],
                            color_discrete_map = {"High Priority require foreign aid": "Red",
                                                  "Medium Priority require financial aid": "Green",
                                                  "Low Priority require no aid": "Yellow"
                                                }
                            )        
        fig.update_layout(legend_title_text="Color Maps");
        fig.show();   
    
    print(wcss)
        

    
    
    
# univariate_eda(data);
# bivariate(data)
# top_sorted_least = least_five_country_child_mortality(data);
# sns.barplot(x="child_mort", y="country", data=top_sorted_least, orient="h", palette="Set2");
# sorted_least = least_five_country_child_mortality(data);
# sns.barplot(x="child_mort", y="country", data=sorted_least, orient="h", palette="Set2");

featured = feature_engineering(data);
modelling(featured);
plt.show();


