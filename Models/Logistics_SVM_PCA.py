import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix
from Import_EDA import load_data

normal_train_images, pneu_train_images, normal_test_images, pneu_test_images = load_data()
X_train = np.concatenate([normal_train_images, pneu_train_images], axis=0)
y_train = np.array([0] * normal_train_images.shape[0] + [1] * pneu_train_images.shape[0])
X_test = np.concatenate([normal_test_images, pneu_test_images], axis=0)
y_test = np.array([0] * normal_test_images.shape[0] + [1] * pneu_test_images.shape[0])

# Fit PCA
pca = PCA()
pca.fit(X_train)
var = pca.explained_variance_ratio_
components = pca.components_
sns.lineplot(np.arange(1, X_train.shape[1] + 1), np.cumsum(var))
plt.title('Percent of Variance Captured vs Number of Principal Component')
plt.savefig('Project/Plot/VarianceCaptured')
plt.show()
sns.lineplot(np.arange(1, 201), np.cumsum(var)[:200])
plt.title('Percent of Variance Captured of the first 200 components')
plt.savefig('Project/Plot/VarianceCaptured_200')
plt.show()

# Transform feature PCA
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Logistic Regression without feature transform
logis_ori = LogisticRegression(solver='saga', max_iter=500)
logis_ori.fit(X_train, y_train)
y_pred = logis_ori.predict(X_test)
print('Train Accuracy: ', metrics.accuracy_score(y_train, logis_ori.predict(X_train)))
print('Test Accuracy: ', metrics.accuracy_score(y_test, y_pred))


# Logistic Regression with feature transform
comp = []
accuracy_train_logis_pca = []
accuracy_test_logis_pca = []
for i in range(1, 200):
    logis_pca = LogisticRegression(solver='saga')
    logis_pca.fit(X_train_pca[:, :i], y_train)
    y_pred_logis_pca = logis_pca.predict(X_test_pca[:, :i])
    accuracy_test_logis_pca.append(metrics.accuracy_score(y_test, y_pred_logis_pca))
    accuracy_train_logis_pca.append(metrics.accuracy_score(y_train, logis_pca.predict(X_train_pca[:, :i])))
    comp.append(i)
sns.lineplot(x=comp[:50], y=accuracy_test_logis_pca[:50])
sns.lineplot(x=comp[:50], y=accuracy_train_logis_pca[:50])
plt.legend(['test accuracy', 'train accuracy'])
plt.title('Logistic Regression Accuracy vs PCA components')
plt.xlabel('Components')
plt.ylabel('Accuracies')
plt.savefig('Project/Plot/Log_PCA_Acc_vs_Comp.png')
plt.show()

# Best Logistic Regression
logis_pca = LogisticRegression(solver='saga')
logis_pca.fit(X_train_pca[:, :11], y_train)
y_pred_logis_pca = logis_pca.predict(X_test_pca[:, :11])
print('Train Accuracy: ', metrics.accuracy_score(y_train, logis_pca.predict(X_train_pca[:, :11])))
print('Test Accuracy: ', metrics.accuracy_score(y_test, y_pred_logis_pca))

## Confusion Metric best Logistic
cf_matrix = confusion_matrix(y_test, y_pred_logis_pca)
con_mat = pd.crosstab(y_test, y_pred_logis_pca, rownames=['Actual'], colnames=['Predicted'])
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ['{:,}'.format(value) for value in cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Logistic Regression with PCA (On Test Set)')
plt.savefig('Project/Plot/Conf_Mat_Log_PCA.png')
plt.show()

# SVM without feature transform
svm_ori = SVC()
svm_ori.fit(X_train, y_train)
y_pred_svm_ori = svm_ori.predict(X_test)
print('Train Accuracy: ', metrics.accuracy_score(y_train, svm_ori.predict(X_train)))
print('Test Accuracy: ', metrics.accuracy_score(y_test, y_pred_svm_ori))


# SVM with feature transform
accuracy_train_svm_pca = []
accuracy_test_svm_pca = []
for i in range(1, 200):
    svm_pca = SVC()
    svm_pca.fit(X_train_pca[:, :i], y_train)
    y_pred_svm_pca = svm_pca.predict(X_test_pca[:, :i])
    accuracy_test_svm_pca.append(metrics.accuracy_score(y_test, y_pred_svm_pca))
    accuracy_train_svm_pca.append(metrics.accuracy_score(y_train, svm_pca.predict(X_train_pca[:, :i])))
sns.lineplot(x=comp[:50], y=accuracy_test_svm_pca[:50])
sns.lineplot(x=comp[:50], y=accuracy_train_svm_pca[:50])
plt.legend(['test accuracy', 'train accuracy'])
plt.title('SVM Accuracy vs PCA components')
plt.xlabel('Components')
plt.ylabel('Accuracies')
plt.savefig('Project/Plot/SVM_PCA_Acc_vs_Comp')
plt.show()

# Best SVM
svm_pca = SVC()
svm_pca.fit(X_train_pca[:, :5], y_train)
y_pred_train_svm_pca = svm_pca.predict(X_train_pca[:, :5])
y_pred_test_svm_pca = svm_pca.predict(X_test_pca[:, :5])
print('Train Accurayc: ', metrics.accuracy_score(y_train, y_pred_train_svm_pca))
print('Test Accuracy: ', metrics.accuracy_score(y_test, y_pred_test_svm_pca))

## Confusion Metric best SVM
cf_matrix = confusion_matrix(y_test, y_pred_test_svm_pca)
con_mat = pd.crosstab(y_test, y_pred_test_svm_pca, rownames=['Actual'], colnames=['Predicted'])
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ['{:,}'.format(value) for value in cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix SVM with PCA (On Test Set)')
plt.savefig('Project/Plot/Conf_Mat_SVM_PCA.png')
plt.show()
