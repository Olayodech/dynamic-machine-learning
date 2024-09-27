import pandas as pd;
import numpy as np;
from matplotlib import pyplot as plt
from sklearn import preprocessing;
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression, SGDClassifier;
from sklearn.linear_model import LogisticRegression;
from sklearn.metrics import mean_squared_error, r2_score;
from sklearn.svm import SVC;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.naive_bayes import GaussianNB;
# from xgboost import XGBClassifier;

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix;
import warnings

# warnings.filterwarnings('ignore');

df = pd.read_csv(r"../../heart - heart.csv");
# univariate analysis
column_data = df.columns;
def age_bracket(age):
    if (age < 35):
        return "Young adult <34";
    elif (age >= 35 and age < 50):
        return "Middle aged adult (35-49)";
    elif (age >= 50 and age < 65):
        return "Senior adult (50-64)";
    else:
        return "Elderly adult";
    
def gender_func(sex):
    if (sex == 1):
        return "Male";
    else:
        return "Female";
    
def chest_pain(pain):
    if (pain == 1):
        return "Typical angina";
    elif (pain == 2):
        return "Atypical angina";
    elif (pain == 3):
        return "Non-anginal pain";
    else:
        return "Asymptomatic";
    
def rest_ecg_calc(ecg):
    if (ecg == 0):
        return "Normal";
    elif (ecg == 1):
        return "ST-T wave abnormality";
    else:
        return "Left ventricular hypertrophy";
    
def target_func(target):
    if (target == 1):
        return "Heart disease";
    else:
        return "No heart disease";


def univariates(df: pd.DataFrame):      
    df["age_bracket"] = df["age"].apply(age_bracket);
    df["gender"] = df["sex"].apply(gender_func);
    df["chest_pain"] = df["cp"].apply(chest_pain);
    df["ecg_test"] = df["restecg"].apply(rest_ecg_calc);
    df["target"] = df["target"].apply(target_func);
    
    fig, axes = plt.subplots(2,4, figsize=(1,1));
    
    sns.countplot(df["ecg_test"], color="green", ax=axes[0,0]);
    sns.countplot(df["chest_pain"], color="yellow", ax=axes[0,1]);
    sns.countplot(df["gender"], color="blue", ax=axes[1,0]);
    sns.countplot(x=df["age_bracket"], color="red", ax=axes[1,1]);
    sns.countplot(x=df["target"], color="red", ax=axes[1,2]);
    axes[1,1].set_title("Patient age group");
    plt.tight_layout()
    plt.show();
    
def bivariates(df: pd.DataFrame):
    df["age_bracket"] = df["age"].apply(age_bracket);
    df["gender"] = df["sex"].apply(gender_func);
    df["chest_pain"] = df["cp"].apply(chest_pain);
    df["ecg_test"] = df["restecg"].apply(rest_ecg_calc);
    df["target"] = df["target"].apply(target_func);
    
    fig, axes = plt.subplots(2,2, figsize=(1,1));
    sns.countplot(x=df["age_bracket"], data=df, color="blue", hue="target", ax=axes[0,0]);
    sns.countplot(x=df["gender"], data=df, color="yellow", hue="target", ax=axes[0,1]);
    plt.tight_layout()
    plt.show();
    
def multivariate(df: pd.DataFrame):
    plt.figure(figsize=(10,10));
    sns.heatmap(df.corr(), cbar=True, annot=True, square=True, cmap='coolwarm');
    # sns.pairplot(df, hue="target");
    plt.show();
    
def feature_engineering(df):
    label = df[['target']]
    df1 = df.drop('target', axis=1);
    encoder = preprocessing.LabelEncoder();
    scaler = MinMaxScaler();
    # normalisation
    df['scaled_restecg'] = scaler.fit_transform(df1['restecg'].values.reshape(-1, 1));
    df['scaled_chol'] = scaler.fit_transform(df1['chol'].values.reshape(-1, 1));
    df['scaled_trestbps'] = scaler.fit_transform(df1['trestbps'].values.reshape(-1, 1));
    df['scaled_tha'] = scaler.fit_transform(df1['thal'].values.reshape(-1, 1));
    df.drop(['restecg', 'oldpeak', 'target', 'thalach', 'cp', 'age', 'sex', 'fbs', 'exang', 'chol', 'slope', 'ca', 'thal', 'trestbps'], axis=1, inplace=True)
    return df, label;
    
def train_data(df: pd.DataFrame, label):
    x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=0.2, random_state=42);
    print(x_test.head(5))
    print(y_test.head(5))
    log_reg = LogisticRegression();
    log_reg.fit(x_train, y_train)
    log_reg_predict = log_reg.predict(x_test)
    mtx = confusion_matrix(y_test, log_reg_predict);
    print("accuracy score : ", accuracy_score(y_test, log_reg_predict))
    print("precision score : ", precision_score(y_test, log_reg_predict))
    print("recall : ", recall_score(y_test, log_reg_predict))
    print("fi score : ", f1_score(y_test, log_reg_predict))
    print("mtx : ", mtx)
    sns.heatmap(mtx, annot=True, fmt='g');
    plt.show();
    
def multi_model_train(df: pd.DataFrame, label):
    x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=0.2, random_state=42);
    print(y_test.shape)
    print(y_train.shape);
    classifiers = [[RandomForestClassifier(), "Random Forest"], 
                   [KNeighborsClassifier(), "K-Nearest Neighbor"],
                #    [SGDClassifier(), "SDG Classigier"],
                   [SVC(), "svc"],
                   [DecisionTreeClassifier(), "Decision Tree"],
                   [GaussianNB(), "Gaussian Naive Bayes"],
                #    [XGBClassifier(), "XGBoost"],
                   [LogisticRegression(), "Logistic Regression"]];
    accuracy_list = {};
    precision_list = {};
    roc_list = {};
    
    # for index, classifier in enumerate(classifiers):
    for classifier in classifiers:
        model = classifier[0];
        model.fit(x_train, y_train);
        model_name = classifier[1];
        
        predictions = model.predict(x_test);
        
        acc_score = accuracy_score(y_test, predictions);
        p_scroe = precision_score(y_test, predictions);
        # recall_score = recall_score(y_test, predictions)
        
        accuracy_list[model_name] = ([str(round(acc_score * 100, 2)) + '%']);
        
        if model_name != classifiers[-1][1]:
            print('')
        
# bivariates(df);
# multivariate(df);
df, label = feature_engineering(df);
# train_data(df, label)
multi_model_train(df, label);

