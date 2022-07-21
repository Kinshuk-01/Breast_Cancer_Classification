import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

    
data = pd.read_csv("D:/KINSHUK/React Course/textutils/Breast_Cancer_Classification/2_breast-cancer.csv")


st.write("""
# Breast Cancer Classification
""")

st.sidebar.title("Data Analysis: ")
option = ["Show Whole Data Set", "Diagnosis Distribution Graph", "Feature Distribution Graphs", "Feautre Correlation Matrix"]
gph_sel = st.sidebar.selectbox("Select an option:", option)

feature = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "symmetry_mean"]
if gph_sel=="Feature Distribution Graphs":
    feat_col = st.sidebar.selectbox("Select a feature to plot:", feature)


st.sidebar.title("TEST THE DATA HERE: ")
if st.sidebar.button("Close"):
    st.write("")
    
    
test_percent = st.sidebar.slider("Select Test Data Percentage:", 10, 50, 20, 5)
clf_name = st.sidebar.selectbox('Select Classifier:', ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM'])


def add_parameter_ui(clf_name):
    para = dict()
    if clf_name == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 1, 100, 100, 10)
        max_depth = st.sidebar.slider("max_depth", 1, 10)
        para["n_estimators_rf"] = n_estimators
        para["max_depth_rf"] = max_depth
    elif clf_name == "Gradient Boosting":
        n_estimators = st.sidebar.slider("n_estimators", 1, 100, 100, 10)
        learning_rate = st.sidebar.slider("learning_rate", 0.1, 1.0, 1.0)
        max_depth = st.sidebar.slider("max_depth", 1, 10)
        para["n_estimators"] = n_estimators
        para["learning_rate"] = learning_rate
        para["max_depth"] = max_depth
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 1.0, 10.0, 2.0, 0.5)
        para["C"] = C
    else:
        pass
    return para



def get_classifier(clf_name, para):
    if clf_name == "Logistic Regression":
        clf = LogisticRegression()
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=para["n_estimators_rf"], max_depth=para["max_depth_rf"])
    elif clf_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(n_estimators=para["n_estimators"], learning_rate=para["learning_rate"], max_depth=para["max_depth"])
    else:
        clf = SVC(C=para["C"], kernel='rbf')
    return clf


para = add_parameter_ui(clf_name)
clf = get_classifier(clf_name, para)

# Dropping Columns having -ve correlation with target column
data = data.drop(['id', 'fractal_dimension_mean', 'texture_se', 'smoothness_se', 'symmetry_se', 'fractal_dimension_se'], axis=1)

# M-1 and B-0
labelencoder= LabelEncoder()
data.iloc[:,0]= labelencoder.fit_transform(data.iloc[:,0].values)


y = data['diagnosis'].values
X = data.drop('diagnosis', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (test_percent/100), random_state=0)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled) 


col1, col2 = st.columns(2)

with col1:
    acc = accuracy_score(y_test, y_pred)
    st.write(f""" 
             ##### Classifier = {clf_name}
             ##### Accuracy = {acc}
    """)
    
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(ax=ax, data=cm, annot=True)
    
    if st.button("Confusion Matrix"):
        st.pyplot(fig, figsize=(2, 2))

with col2:
    row, col = data.shape
    st.write(f"""
             ##### Number of Rows: {str(row)}
             ##### Number of Columns: {str(col)}
    """)
    
    if st.button("Data Description"):
        st.write(data.describe())

st.set_option('deprecation.showPyplotGlobalUse', False)
# Plot Features
st.header("Plotting Graph:")

M = data[(data['diagnosis'] != 0)]
B = data[(data['diagnosis'] == 0)]

def plot_distribution(data_select, size_bin) :  
    tmp1 = M[data_select]
    tmp2 = B[data_select]
    hist_data = [tmp1, tmp2]
    
    group_labels = ['malignant', 'benign']
    colors = ['#FFD700', '#7EC0EE']

    figu, ax = plt.subplots()
    plt.hist(hist_data, color=colors)
    plt.title(data_select)
    plt.xlabel(data_select + " value")
    plt.ylabel("No. of Patients")
    plt.legend(group_labels)


def add_graph(gph_sel):
    if gph_sel=="Show Whole Data Set":
        st.write(pd.DataFrame(data))
        
    elif gph_sel=="Diagnosis Distribution Graph":
        pie_cht, ax = plt.subplots()
        y = data['diagnosis'].value_counts()/100
        mylabels = ['benign', 'malignent']
        mycolors = ['#FFD700', '#7EC0EE']
        myexplode = [0.2, 0]
        plt.pie(y, labels = mylabels, autopct='%1.1f%%', colors = mycolors, explode = myexplode, shadow = True)
        plt.legend()
        st.pyplot(pie_cht)
        
    elif gph_sel=="Feature Distribution Graphs":
        plott = plot_distribution(feat_col, .5)
        st.pyplot(plott)
        
    else:
        corr = data.corr()
        fig_corr, ax = plt.subplots()
        sns.heatmap(data=corr, annot=False)
        st.pyplot(fig_corr)
    
add_graph(gph_sel)
