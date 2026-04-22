#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve , confusion_matrix , classification_report
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA


# # DATA PREPROCESSING

# In[3]:


train_real = pd.read_csv("Wav2Vec2_Crema_dev_fake.csv")
train_fake = pd.read_csv("Wav2Vec2_Crema_dev_real.csv")
dev_real = pd.read_csv("Wav2Vec2_Crema_test_fake.csv")
dev_fake = pd.read_csv("Wav2Vec2_Crema_test_real.csv")
test_real = pd.read_csv("Wav2Vec2_Crema_train_fake.csv")
test_fake = pd.read_csv("Wav2Vec2_Crema_train_real.csv")


# In[4]:


train_real["label"] , train_fake["label"] = 1, 0
dev_real["label"] , dev_fake["label"] = 1 , 0
test_real["label"] , test_fake["label"] = 1 , 0

train_data = pd.concat([train_real, train_fake], ignore_index=True)
val_data = pd.concat([dev_real, dev_fake], ignore_index=True)
test_data = pd.concat([test_real, test_fake], ignore_index=True)

train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
val_data = val_data.sample(frac=1, random_state=42).reset_index(drop=True)
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

non_feature_columns = ["label", train_data.columns[0]]  
X_train = train_data.drop(columns=non_feature_columns)
y_train = train_data["label"]
X_val = val_data.drop(columns=non_feature_columns)
y_val = val_data["label"]
X_test = test_data.drop(columns=non_feature_columns)
y_test = test_data["label"]


# In[5]:


X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_val = X_val.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train_full = np.vstack((X_train, X_val))
y_train_full = np.hstack((y_train, y_val))

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

pca = PCA(n_components=100)
X_train_full_pca = pca.fit_transform(X_train_full)
X_test_pca = pca.transform(X_test)

X_train_pca, X_val_pca, y_train, y_val = train_test_split(
    X_train_full_pca, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)


# In[6]:


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()


def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer


# In[40]:


labels, counts = np.unique(y_train, return_counts=True)

plt.pie(counts, labels=['Real', 'Fake'], autopct='%0.1f%%', radius=1)
plt.title("Percentage of each class")
plt.show()

labels, counts = np.unique(y_test, return_counts=True)

plt.pie(counts, labels=['Real', 'Fake'], autopct='%0.1f%%', radius=1)
plt.title("Percentage of each class")
plt.show()


# # NEURAL NETWORK

# In[7]:


mlp_model = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", solver="adam", max_iter=500, random_state=42)
mlp_pca_model = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", solver="adam", max_iter=500, random_state=42)

mlp_model.fit(X_train, y_train)
mlp_pca_model.fit(X_train_pca,y_train)

mlp_pred = mlp_model.predict(X_test)
mlp_probs = mlp_model.predict_proba(X_test)[:, 1]

mlp_pca_pred = mlp_pca_model.predict(X_test_pca)
mlp_pca_probs = mlp_pca_model.predict_proba(X_test_pca)[:, 1]

mlp_results = {
    "Accuracy": accuracy_score(y_test, mlp_pred),
    "Precision": precision_score(y_test, mlp_pred),
    "Recall": recall_score(y_test, mlp_pred),
    "F1-score": f1_score(y_test, mlp_pred),
    "EER": compute_eer(y_test, mlp_probs),
}

mlp_pca_results = {
    "Accuracy": accuracy_score(y_test, mlp_pca_pred),
    "Precision": precision_score(y_test, mlp_pca_pred),
    "Recall": recall_score(y_test, mlp_pca_pred),
    "F1-score": f1_score(y_test, mlp_pca_pred),
    "EER": compute_eer(y_test, mlp_pca_probs),
}

mlp_class_report = classification_report(y_test, mlp_pred, target_names=["Class 0", "Class 1"], output_dict=True)
mlp_class_report_df = pd.DataFrame(mlp_class_report).transpose()
mlp_filtered = mlp_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

mlp_results_df = pd.DataFrame([mlp_results], index=["Neural Network (MLP)"])
print("\nPerformance metrics of Neural Network (MLP):")
print(mlp_results_df)
print(mlp_filtered)

mlp_pca_class_report = classification_report(y_test, mlp_pca_pred, target_names=["Class 0", "Class 1"], output_dict=True)
mlp_pca_class_report_df = pd.DataFrame(mlp_pca_class_report).transpose()
mlp_pca_filtered = mlp_pca_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

mlp_pca_results_df = pd.DataFrame([mlp_pca_results], index=["Neural Network (MLP) (PCA)"])
print("\nPerformance metrics of Neural Network (MLP) (PCA):")
print(mlp_pca_results_df)
print(mlp_pca_filtered)


plot_confusion_matrix(y_test, mlp_pred, "Neural Network (MLP)")
plot_confusion_matrix(y_test,mlp_pca_pred,"Neural Network (MLP) (PCA)")


# # NAIVE BAYES

# In[8]:


nb_model = GaussianNB()
nb_pca_model = GaussianNB()

nb_model.fit(X_train, y_train)
nb_pca_model.fit(X_train_pca,y_train)

nb_pred = nb_model.predict(X_test)
nb_probs = nb_model.predict_proba(X_test)[:, 1]

nb_pca_pred = nb_pca_model.predict(X_test_pca)
nb_pca_probs = nb_pca_model.predict_proba(X_test_pca)[:, 1]

nb_results = {
    "Accuracy": accuracy_score(y_test, nb_pred),
    "Precision": precision_score(y_test, nb_pred),
    "Recall": recall_score(y_test, nb_pred),
    "F1-score": f1_score(y_test, nb_pred),
    "EER": compute_eer(y_test, nb_probs),
}

nb_pca_results = {
    "Accuracy": accuracy_score(y_test, nb_pca_pred),
    "Precision": precision_score(y_test, nb_pca_pred),
    "Recall": recall_score(y_test, nb_pca_pred),
    "F1-score": f1_score(y_test, nb_pca_pred),
    "EER": compute_eer(y_test, nb_pca_probs),
}

nb_class_report = classification_report(y_test, nb_pred, target_names=["Class 0", "Class 1"], output_dict=True)
nb_class_report_df = pd.DataFrame(nb_class_report).transpose()
nb_filtered = nb_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

nb_results_df = pd.DataFrame([nb_results], index=["Naïve Bayes"])
print("\nPerformance metrics of Naïve Bayes:")
print(nb_results_df)
print(nb_filtered)

nb_pca_class_report = classification_report(y_test, nb_pca_pred, target_names=["Class 0", "Class 1"], output_dict=True)
nb_pca_class_report_df = pd.DataFrame(nb_pca_class_report).transpose()
nb_pca_filtered = nb_pca_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

nb_pca_results_df = pd.DataFrame([nb_pca_results], index=["Naïve Bayes (PCA)"])
print("\nPerformance metrics of Naïve Bayes (PCA):")
print(nb_pca_results_df)
print(nb_pca_filtered)

plot_confusion_matrix(y_test, nb_pred, "Naïve Bayes")
plot_confusion_matrix(y_test,nb_pca_pred, "Naive Bayes (PCA)")



# # RANDOM FOREST CLASSIFIER

# In[9]:


rfc_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rfc_pca_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

rfc_model.fit(X_train, y_train)
rfc_pca_model.fit(X_train_pca,y_train)

rfc_pred = rfc_model.predict(X_test)
rfc_probs = rfc_model.predict_proba(X_test)[:, 1]

rfc_pca_pred = rfc_pca_model.predict(X_test_pca)
rfc_pca_probs = rfc_pca_model.predict_proba(X_test_pca)[:, 1]

rfc_results = {
    "Accuracy": accuracy_score(y_test, rfc_pred),
    "Precision": precision_score(y_test, rfc_pred),
    "Recall": recall_score(y_test, rfc_pred),
    "F1-score": f1_score(y_test, rfc_pred),
    "EER": compute_eer(y_test, rfc_probs),
}

rfc_pca_results = {
    "Accuracy": accuracy_score(y_test, rfc_pca_pred),
    "Precision": precision_score(y_test, rfc_pca_pred),
    "Recall": recall_score(y_test, rfc_pca_pred),
    "F1-score": f1_score(y_test, rfc_pca_pred),
    "EER": compute_eer(y_test, rfc_pca_probs),
}

rfc_class_report = classification_report(y_test, rfc_pred, target_names=["Class 0", "Class 1"], output_dict=True)
rfc_pca_class_report = classification_report(y_test, rfc_pca_pred, target_names=["Class 0", "Class 1"], output_dict=True)

rfc_class_report_df = pd.DataFrame(rfc_class_report).transpose()
rfc_pca_class_report_df = pd.DataFrame(rfc_pca_class_report).transpose()

rfc_filtered = rfc_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]
rfc_pca_filtered = rfc_pca_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

rfc_results_df = pd.DataFrame([rfc_results], index=["Random Forest"])
rfc_pca_results_df = pd.DataFrame([rfc_pca_results], index=["Random Forest (PCA)"])

print(f"performance metrics of RFC Classifier : ")
print(rfc_results_df)
print(rfc_filtered)
print(f"performance metrics of RFC Classifier (PCA) : ")
print(rfc_pca_results_df)
print(rfc_pca_filtered)

plot_confusion_matrix(y_test, rfc_pred, "Random Forest")
plot_confusion_matrix(y_test, rfc_pca_pred, "Random Forest (PCA)")


# # XG BOOST

# In[21]:


xgb_model = XGBClassifier()
xgb_pca_model = XGBClassifier()

xgb_model.fit(X_train, y_train)
xgb_pca_model.fit(X_train_pca, y_train)

xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred = xgb_model.predict(X_test)

xgb_pca_probs = xgb_pca_model.predict_proba(X_test_pca)[:, 1]
xgb_pca_pred = xgb_pca_model.predict(X_test_pca)

xgb_results = {
    "Accuracy": accuracy_score(y_test, xgb_pred),
    "Precision": precision_score(y_test, xgb_pred),
    "Recall": recall_score(y_test, xgb_pred),
    "F1-score": f1_score(y_test, xgb_pred),
    "EER": compute_eer(y_test, xgb_probs),
}

xgb_pca_results = {
    "Accuracy": accuracy_score(y_test, xgb_pca_pred),
    "Precision": precision_score(y_test, xgb_pca_pred),
    "Recall": recall_score(y_test, xgb_pca_pred),
    "F1-score": f1_score(y_test, xgb_pca_pred),
    "EER": compute_eer(y_test, xgb_pca_probs),
}

xgb_class_report = classification_report(y_test, xgb_pred, target_names=["Class 0", "Class 1"], output_dict=True)
xgb_class_report_df = pd.DataFrame(xgb_class_report).transpose()
xgb_filtered = xgb_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

xgb_results_df = pd.DataFrame([xgb_results], index=["XGBoost"])
print("\nPerformance metrics of XGBoost:")
print(xgb_results_df)
print(xgb_filtered)

xgb_pca_class_report = classification_report(y_test, xgb_pca_pred, target_names=["Class 0", "Class 1"], output_dict=True)
xgb_pca_class_report_df = pd.DataFrame(xgb_pca_class_report).transpose()
xgb_pca_filtered = xgb_pca_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

xgb_pca_results_df = pd.DataFrame([xgb_pca_results], index=["XGBoost (PCA)"])
print("\nPerformance metrics of XGBoost (PCA):")
print(xgb_pca_results_df)
print(xgb_pca_filtered)

plot_confusion_matrix(y_test, xgb_pred, "XG Boost")
plot_confusion_matrix(y_test, xgb_pca_pred, "XG Boost (PCA)")


# # PERCEPTRON MODEL

# In[31]:


perc_model = Perceptron(max_iter=1000, tol=1e-3)
perc_pca_model = Perceptron(max_iter=1000, tol=1e-3)

perc_model.fit(X_train, y_train)
perc_pca_model.fit(X_train_pca, y_train)

perc_pred = perc_model.predict(X_test)
perc_decision_scores = perc_model.decision_function(X_test)
perc_probs = (perc_decision_scores - perc_decision_scores.min()) / (perc_decision_scores.max() - perc_decision_scores.min())

perc_pca_pred = perc_pca_model.predict(X_test_pca)
perc_pca_decision_scores = perc_pca_model.decision_function(X_test_pca)
perc_pca_probs = (perc_pca_decision_scores - perc_pca_decision_scores.min()) / (perc_pca_decision_scores.max() - perc_pca_decision_scores.min())

perc_results = {
    "Accuracy": accuracy_score(y_test, perc_pred),
    "Precision": precision_score(y_test, perc_pred),
    "Recall": recall_score(y_test, perc_pred),
    "F1-score": f1_score(y_test, perc_pred),
    "EER": compute_eer(y_test, perc_probs)
}

perc_pca_results = {
    "Accuracy": accuracy_score(y_test, perc_pca_pred),
    "Precision": precision_score(y_test, perc_pca_pred),
    "Recall": recall_score(y_test, perc_pca_pred),
    "F1-score": f1_score(y_test, perc_pca_pred),
    "EER": compute_eer(y_test, perc_pca_probs)
}

perc_class_report = classification_report(y_test, perc_pred, target_names=["Class 0", "Class 1"], output_dict=True)
perc_class_report_df = pd.DataFrame(perc_class_report).transpose()
perc_filtered = perc_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

perc_results_df = pd.DataFrame([perc_results], index=["Perceptron"])
print("\nPerformance metrics of Perceptron:")
print(perc_results_df)
print(perc_filtered)

perc_pca_class_report = classification_report(y_test, perc_pca_pred, target_names=["Class 0", "Class 1"], output_dict=True)
perc_pca_class_report_df = pd.DataFrame(perc_pca_class_report).transpose()
perc_pca_filtered = perc_pca_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

perc_pca_results_df = pd.DataFrame([perc_pca_results], index=["Perceptron (PCA)"])
print("\nPerformance metrics of Perceptron (PCA):")
print(perc_pca_results_df)
print(perc_pca_filtered)

plot_confusion_matrix(y_test, perc_pred, "Perceptron")
plot_confusion_matrix(y_test, perc_pca_pred, "Perceptron (PCA)")


# # LOGISTIC MODEL

# In[32]:


log_model = LogisticRegression(max_iter=1000)
log_pca_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_probs = log_model.predict_proba(X_test)[:, 1]

log_pca_model.fit(X_train_pca, y_train)
log_pca_pred = log_pca_model.predict(X_test_pca)
log_pca_probs = log_pca_model.predict_proba(X_test_pca)[:, 1]

log_results = {
    "Accuracy": accuracy_score(y_test, log_pred),
    "Precision": precision_score(y_test, log_pred),
    "Recall": recall_score(y_test, log_pred),
    "F1-score": f1_score(y_test, log_pred),
    "EER": compute_eer(y_test, log_probs)
}

log_pca_results = {
    "Accuracy": accuracy_score(y_test, log_pca_pred),
    "Precision": precision_score(y_test, log_pca_pred),
    "Recall": recall_score(y_test, log_pca_pred),
    "F1-score": f1_score(y_test, log_pca_pred),
    "EER": compute_eer(y_test, log_pca_probs)
}

log_class_report = classification_report(y_test, log_pred, target_names=["Class 0", "Class 1"], output_dict=True)
log_class_report_df = pd.DataFrame(log_class_report).transpose()
log_filtered = log_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

log_results_df = pd.DataFrame([log_results], index=["Logistic"])
print("\nPerformance metrics of Logistic Regression:")
print(log_results_df)
print(log_filtered)

log_pca_class_report = classification_report(y_test, log_pca_pred, target_names=["Class 0", "Class 1"], output_dict=True)
log_pca_class_report_df = pd.DataFrame(log_pca_class_report).transpose()
log_pca_filtered = log_pca_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

log_pca_results_df = pd.DataFrame([log_pca_results], index=["Logistic (PCA)"])
print("\nPerformance metrics of Logistic Regression (PCA):")
print(log_pca_results_df)
print(log_pca_filtered)

plot_confusion_matrix(y_test, log_pred, "Logistic")
plot_confusion_matrix(y_test, log_pca_pred, "Logistic (PCA)")


# # SVM MODEL

# Linear Kernel

# In[16]:


SVM_linear = SVC(kernel="linear", C=1.0, class_weight="balanced")
SVM_linear_pca = SVC(kernel="linear", C=1.0)

SVM_linear.fit(X_train, y_train)
SVM_linear_pca.fit(X_train_pca,y_train)

svm_linear_pred = SVM_linear.predict(X_test)
svm_linear_pca_pred = SVM_linear_pca.predict(X_test_pca)

decision_scores_linear = SVM_linear.decision_function(X_test)
probs_linear = (decision_scores_linear - decision_scores_linear.min()) / (decision_scores_linear.max() - decision_scores_linear.min())

decision_scores_linear_pca = SVM_linear_pca.decision_function(X_test_pca)
probs_pca_linear = (decision_scores_linear_pca - decision_scores_linear_pca.min()) / (decision_scores_linear_pca.max() - decision_scores_linear_pca.min())

svm_linear_results = {
    "Accuracy": accuracy_score(y_test, svm_linear_pred),
    "Precision": precision_score(y_test, svm_linear_pred),
    "Recall": recall_score(y_test, svm_linear_pred),
    "F1-score": f1_score(y_test, svm_linear_pred),
    "EER": compute_eer(y_test, probs_linear),
}

svm_pca_linear_results = {
    "Accuracy": accuracy_score(y_test, svm_linear_pca_pred),
    "Precision": precision_score(y_test, svm_linear_pca_pred),
    "Recall": recall_score(y_test, svm_linear_pca_pred),
    "F1-score": f1_score(y_test, svm_linear_pca_pred),
    "EER": compute_eer(y_test, probs_pca_linear),
}

svm_linear_class_report = classification_report(y_test, svm_linear_pred, target_names=["Class 0", "Class 1"], output_dict=True)
svm_linear_class_report_df = pd.DataFrame(svm_linear_class_report).transpose()
svm_linear_filtered = svm_linear_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

svm_linear_results_df = pd.DataFrame([svm_linear_results], index=["SVM Linear"])
print("\nPerformance metrics of SVM Linear:")
print(svm_linear_results_df)
print(svm_linear_filtered)

svm_pca_linear_class_report = classification_report(y_test, svm_linear_pca_pred, target_names=["Class 0", "Class 1"], output_dict=True)
svm_pca_linear_class_report_df = pd.DataFrame(svm_pca_linear_class_report).transpose()
svm_pca_linear_filtered = svm_pca_linear_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

svm_pca_linear_results_df = pd.DataFrame([svm_pca_linear_results], index=["SVM Linear (PCA)"])
print("\nPerformance metrics of SVM Linear (PCA):")
print(svm_pca_linear_results_df)
print(svm_pca_linear_filtered)

plot_confusion_matrix(y_test, svm_linear_pred, "SVM Linear")
plot_confusion_matrix(y_test, svm_linear_pca_pred, "SVM Linear (PCA)")



# RBF Kernel

# In[17]:


SVM_rbf = SVC(kernel="rbf", C=1.0, gamma=0.001, class_weight="balanced")
SVM_rbf_pca = SVC(kernel="rbf", C=1.0, gamma=0.001, class_weight="balanced")

SVM_rbf.fit(X_train, y_train)
SVM_rbf_pca.fit(X_train_pca, y_train)

svm_rbf_pred = SVM_rbf.predict(X_test)
svm_rbf_pca_pred = SVM_rbf_pca.predict(X_test_pca)

decision_scores_rbf = SVM_rbf.decision_function(X_test)
probs_rbf = (decision_scores_rbf - decision_scores_rbf.min()) / (decision_scores_rbf.max() - decision_scores_rbf.min())

decision_scores_rbf_pca = SVM_rbf_pca.decision_function(X_test_pca)
probs_pca_rbf = (decision_scores_rbf_pca - decision_scores_rbf_pca.min()) / (decision_scores_rbf_pca.max() - decision_scores_rbf_pca.min())

svm_rbf_results = {
    "Accuracy": accuracy_score(y_test, svm_rbf_pred),
    "Precision": precision_score(y_test, svm_rbf_pred),
    "Recall": recall_score(y_test, svm_rbf_pred),
    "F1-score": f1_score(y_test, svm_rbf_pred),
    "EER": compute_eer(y_test, probs_rbf),
}

svm_pca_rbf_results = {
    "Accuracy": accuracy_score(y_test, svm_rbf_pca_pred),
    "Precision": precision_score(y_test, svm_rbf_pca_pred),
    "Recall": recall_score(y_test, svm_rbf_pca_pred),
    "F1-score": f1_score(y_test, svm_rbf_pca_pred),
    "EER": compute_eer(y_test, probs_pca_rbf),
}

svm_rbf_class_report = classification_report(y_test, svm_rbf_pred, target_names=["Class 0", "Class 1"], output_dict=True)
svm_rbf_class_report_df = pd.DataFrame(svm_rbf_class_report).transpose()
svm_rbf_filtered = svm_rbf_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

svm_rbf_results_df = pd.DataFrame([svm_rbf_results], index=["SVM RBF"])
print("\nPerformance metrics of SVM RBF:")
print(svm_rbf_results_df)
print(svm_rbf_filtered)

svm_pca_rbf_class_report = classification_report(y_test, svm_rbf_pca_pred, target_names=["Class 0", "Class 1"], output_dict=True)
svm_pca_rbf_class_report_df = pd.DataFrame(svm_pca_rbf_class_report).transpose()
svm_pca_rbf_filtered = svm_pca_rbf_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

svm_pca_rbf_results_df = pd.DataFrame([svm_pca_rbf_results], index=["SVM RBF (PCA)"])
print("\nPerformance metrics of SVM RBF (PCA):")
print(svm_pca_rbf_results_df)
print(svm_pca_rbf_filtered)

plot_confusion_matrix(y_test, svm_rbf_pred, "SVM RBF")
plot_confusion_matrix(y_test, svm_rbf_pca_pred, "SVM RBF (PCA)")


# Polynomial Kernel

# In[18]:


SVM_poly = SVC(kernel="poly", C=1.0, class_weight="balanced")
SVM_poly_pca = SVC(kernel="poly", C=1.0)

SVM_poly.fit(X_train, y_train)
SVM_poly_pca.fit(X_train_pca, y_train)

svm_poly_pred = SVM_poly.predict(X_test)
svm_poly_pca_pred = SVM_poly_pca.predict(X_test_pca)

decision_scores_poly = SVM_poly.decision_function(X_test)
probs_poly = (decision_scores_poly - decision_scores_poly.min()) / (decision_scores_poly.max() - decision_scores_poly.min())

decision_scores_poly_pca = SVM_poly_pca.decision_function(X_test_pca)
probs_pca_poly = (decision_scores_poly_pca - decision_scores_poly_pca.min()) / (decision_scores_poly_pca.max() - decision_scores_poly_pca.min())

svm_poly_results = {
    "Accuracy": accuracy_score(y_test, svm_poly_pred),
    "Precision": precision_score(y_test, svm_poly_pred),
    "Recall": recall_score(y_test, svm_poly_pred),
    "F1-score": f1_score(y_test, svm_poly_pred),
    "EER": compute_eer(y_test, probs_poly),
}

svm_pca_poly_results = {
    "Accuracy": accuracy_score(y_test, svm_poly_pca_pred),
    "Precision": precision_score(y_test, svm_poly_pca_pred),
    "Recall": recall_score(y_test, svm_poly_pca_pred),
    "F1-score": f1_score(y_test, svm_poly_pca_pred),
    "EER": compute_eer(y_test, probs_pca_poly),
}

svm_poly_class_report = classification_report(y_test, svm_poly_pred, target_names=["Class 0", "Class 1"], output_dict=True)
svm_poly_class_report_df = pd.DataFrame(svm_poly_class_report).transpose()
svm_poly_filtered = svm_poly_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

svm_poly_results_df = pd.DataFrame([svm_poly_results], index=["SVM Polynomial"])
print("\nPerformance metrics of SVM POlynomial:")
print(svm_poly_results_df)
print(svm_poly_filtered)

svm_pca_poly_class_report = classification_report(y_test, svm_poly_pca_pred, target_names=["Class 0", "Class 1"], output_dict=True)
svm_pca_poly_class_report_df = pd.DataFrame(svm_pca_poly_class_report).transpose()
svm_pca_poly_filtered = svm_pca_poly_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]

svm_pca_poly_results_df = pd.DataFrame([svm_pca_poly_results], index=["SVM Polynomial (PCA)"])
print("\nPerformance metrics of SVM Polynomial (PCA):")
print(svm_pca_poly_results_df)
print(svm_pca_poly_filtered)

plot_confusion_matrix(y_test, svm_poly_pred, "SVM Polynomial")
plot_confusion_matrix(y_test, svm_poly_pca_pred, "SVM Polynomial (PCA)")


# # KNN CLASSIFIER

# In[ ]:


knn_results_list = []
knn_pca_results_list = []
k_values = [1, 7, 11, 13, 17, 19, 23, 29, 31]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    y_prob_knn = knn.predict_proba(X_test)[:, 1]  
    
    knn_results = {
        "Accuracy": accuracy_score(y_test, y_pred_knn),
        "Precision": precision_score(y_test, y_pred_knn),
        "Recall": recall_score(y_test, y_pred_knn),
        "F1-score": f1_score(y_test, y_pred_knn),
        "EER": compute_eer(y_test, y_prob_knn),
    }
    
    knn_class_report = classification_report(y_test, y_pred_knn, target_names=["Class 0", "Class 1"], output_dict=True)
    knn_class_report_df = pd.DataFrame(knn_class_report).transpose()
    knn_filtered = knn_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]
    
    knn_results_list.append((k, knn_results, knn_filtered))
    
    knn_pca = KNeighborsClassifier(n_neighbors=k)
    knn_pca.fit(X_train_pca, y_train)
    y_pred_knn_pca = knn_pca.predict(X_test_pca)
    y_prob_knn_pca = knn_pca.predict_proba(X_test_pca)[:, 1] 
    
    knn_pca_results = {
        "Accuracy": accuracy_score(y_test, y_pred_knn_pca),
        "Precision": precision_score(y_test, y_pred_knn_pca),
        "Recall": recall_score(y_test, y_pred_knn_pca),
        "F1-score": f1_score(y_test, y_pred_knn_pca),
        "EER": compute_eer(y_test, y_prob_knn_pca),
    }
    
    knn_pca_class_report = classification_report(y_test, y_pred_knn_pca, target_names=["Class 0", "Class 1"], output_dict=True)
    knn_pca_class_report_df = pd.DataFrame(knn_pca_class_report).transpose()
    knn_pca_filtered = knn_pca_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]
    
    knn_pca_results_list.append((k, knn_pca_results, knn_pca_filtered))

for k, knn_results, knn_filtered in knn_results_list:
    knn_results_df = pd.DataFrame([knn_results], index=[f"KNN (k={k})"])
    print(f"\nPerformance metrics of KNN (k={k}):")
    print(knn_results_df)
    print(knn_filtered)

for k, knn_pca_results, knn_pca_filtered in knn_pca_results_list:
    knn_pca_results_df = pd.DataFrame([knn_pca_results], index=[f"KNN (PCA, k={k})"])
    print(f"\nPerformance metrics of KNN with PCA (k={k}):")
    print(knn_pca_results_df)
    print(knn_pca_filtered)


# In[24]:


optimal_k = 17
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_prob_knn = knn.predict_proba(X_test)[:, 1]  
    
knn_results = {
    "Accuracy": accuracy_score(y_test, y_pred_knn),
    "Precision": precision_score(y_test, y_pred_knn),
    "Recall": recall_score(y_test, y_pred_knn),
    "F1-score": f1_score(y_test, y_pred_knn),
    "EER": compute_eer(y_test, y_prob_knn),
}
    
knn_class_report = classification_report(y_test, y_pred_knn, target_names=["Class 0", "Class 1"], output_dict=True)
knn_class_report_df = pd.DataFrame(knn_class_report).transpose()
knn_filtered = knn_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]
    
knn_pca = KNeighborsClassifier(n_neighbors=optimal_k)
knn_pca.fit(X_train_pca, y_train)
y_pred_knn_pca = knn_pca.predict(X_test_pca)
y_prob_knn_pca = knn_pca.predict_proba(X_test_pca)[:, 1] 
    
knn_pca_results = {
    "Accuracy": accuracy_score(y_test, y_pred_knn_pca),
    "Precision": precision_score(y_test, y_pred_knn_pca),
    "Recall": recall_score(y_test, y_pred_knn_pca),
    "F1-score": f1_score(y_test, y_pred_knn_pca),
    "EER": compute_eer(y_test, y_prob_knn_pca),
}
    
knn_pca_class_report = classification_report(y_test, y_pred_knn_pca, target_names=["Class 0", "Class 1"], output_dict=True)
knn_pca_class_report_df = pd.DataFrame(knn_pca_class_report).transpose()
knn_pca_filtered = knn_pca_class_report_df.loc[["Class 0", "Class 1"], ["precision", "recall", "f1-score"]]
    
knn_results_df = pd.DataFrame([knn_results], index=["KNN (K=17)"])
print("\nPerformance metrics of KNN :")
print(knn_results_df)
print(knn_filtered)

knn_pca_results_df = pd.DataFrame([knn_pca_results], index=["KNN (K=17) (PCA)"])
print("\nPerformance metrics of KNN (PCA):")
print(knn_pca_results_df)
print(knn_pca_filtered)

plot_confusion_matrix(y_test, y_pred_knn, "KNN")
plot_confusion_matrix(y_test, y_pred_knn_pca, "KNN (PCA)")


# # COMPARSION TABLE

# In[33]:


comparison_df = pd.DataFrame(
    [mlp_results, nb_results, rfc_results, xgb_results, perc_results, log_results, svm_linear_results, svm_poly_results, svm_rbf_results, knn_results],
    columns=["Accuracy", "Precision", "Recall", "F1-score", "EER"]
)

comparison_df["Classifiers"]=[
        "Neural Network (MLP)", "Naïve Bayes", "Random Forest", "XG Boost", "Perceptron", "Logistic", "SVM Linear", "SVM Polynomial", "SVM RBF",
        "KNN (k=17)"]

print("Comparison of Classifier Performance :")
comparison_df


# In[34]:


comparison_pca_df = pd.DataFrame(
    [
        mlp_pca_results, nb_pca_results, rfc_pca_results, xgb_pca_results, perc_pca_results, log_pca_results, svm_pca_linear_results, svm_pca_poly_results, 
        svm_pca_rbf_results, knn_pca_results],
    columns=["Accuracy", "Precision", "Recall", "F1-score", "EER"]
)
comparison_pca_df["Classifiers"]=[
        "Neural Network (MLP)", "Naïve Bayes", "Random Forest", "XG Boost", "Perceptron", "Logistic", "SVM Linear", "SVM Polynomial", "SVM RBF",
        "KNN (k=17)"]

comparison_pca_df


# # ROC CURVE 

# In[35]:


fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_probs)
auc_nb = auc(fpr_nb, tpr_nb)
print(len(fpr_nb))
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp_probs)
auc_mlp = auc(fpr_mlp, tpr_mlp)

fpr_rfc, tpr_rfc, _ = roc_curve(y_test, rfc_probs)
auc_rfc = auc(fpr_rfc, tpr_rfc)

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
auc_xgb = auc(fpr_xgb, tpr_xgb)

fpr_perc, tpr_perc, _ = roc_curve(y_test, perc_probs)
auc_perc = auc(fpr_perc, tpr_perc)

fpr_log, tpr_log, _ = roc_curve(y_test, log_probs)
auc_log = auc(fpr_log, tpr_log)

fpr_svm_linear, tpr_svm_linear, _ = roc_curve(y_test, probs_linear)
auc_svm_linear = auc(fpr_svm_linear, tpr_svm_linear)

fpr_svm_rbf, tpr_svm_rbf, _ = roc_curve(y_test, probs_rbf)
auc_svm_rbf = auc(fpr_svm_rbf, tpr_svm_rbf)

fpr_svm_poly, tpr_svm_poly, _ = roc_curve(y_test, probs_poly)
auc_svm_poly = auc(fpr_svm_poly, tpr_svm_poly)

plt.figure(figsize=(10, 8))
plt.plot(fpr_nb, tpr_nb, label=f'Naïve Bayes (AUC = {auc_nb:.2f})', linestyle='--')
plt.plot(fpr_mlp, tpr_mlp, label=f'Neural Network (AUC = {auc_mlp:.2f})', linestyle='--')
plt.plot(fpr_rfc, tpr_rfc, label=f'Random Forest (AUC = {auc_rfc:.2f})', linestyle='--')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.2f})', linestyle='--')
plt.plot(fpr_perc, tpr_perc, label=f'Perceptron (AUC = {auc_perc:.2f})', linestyle='--')
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc_log:.2f})', linestyle='--')
plt.plot(fpr_svm_linear, tpr_svm_linear, label=f'SVM_Linear (AUC = {auc_svm_linear:.2f})', linestyle='--')
plt.plot(fpr_svm_rbf, tpr_svm_rbf, label=f'SVM_RBF (AUC = {auc_svm_rbf:.2f})', linestyle='--')
plt.plot(fpr_svm_poly, tpr_svm_poly, label=f'SVM_Polynomial (AUC = {auc_svm_poly:.2f})', linestyle='--')
#plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# # BAR CHART

# In[37]:


metrics_df = pd.DataFrame(
    [
        mlp_results, nb_results, rfc_results, xgb_results, 
        perc_results, log_results, svm_linear_results, 
        svm_poly_results, svm_rbf_results, knn_results
    ],
    index=[
        "Neural Network (MLP)", "Naïve Bayes", "Random Forest", 
        "XG Boost", "Perceptron", "Logistic", 
        "SVM Linear", "SVM Polynomial", "SVM RBF", "KNN"
    ]
)

for metric in metrics_df.columns:
    metrics_df[metric].plot(kind="bar", figsize=(12, 6), edgecolor='black')
    plt.title(f"Performance Comparison of Classifiers - {metric}")
    plt.ylabel(f"{metric} Score")
    plt.xlabel("Classifiers")
    plt.xticks(rotation=0)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


# In[39]:


metrics_df = pd.DataFrame(
    [
        mlp_pca_results, nb_pca_results, rfc_pca_results, xgb_pca_results, perc_pca_results, log_pca_results, svm_pca_linear_results, svm_pca_poly_results, 
        svm_pca_rbf_results, knn_pca_results
    ],
    index=[
        "Neural Network (MLP)", "Naïve Bayes", "Random Forest", 
        "XG Boost", "Perceptron", "Logistic", 
        "SVM Linear", "SVM Polynomial", "SVM RBF", "KNN"
    ]
)

for metric in metrics_df.columns:
    metrics_df[metric].plot(kind="bar", figsize=(12, 6), edgecolor='black')
    plt.title(f"Performance Comparison of Classifiers (PCA) - {metric}")
    plt.ylabel(f"{metric} Score")
    plt.xlabel("Classifiers")
    plt.xticks(rotation=0)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

