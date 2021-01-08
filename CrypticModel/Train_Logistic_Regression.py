import pandas as pd
import csv
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import pickle

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

dominant_testing = sys.argv[1]
dominant_training = sys.argv[2]
cryptic_training = sys.argv[3]
cryptic_testing = sys.argv[4]

C_ = 0.0018

pict_name_rocauc = 'lr_rocauc.png'
pict_name_prerec = 'lr_prerec.png'
importances_output = 'feat_importances.csv'
model_name = 'finalized_model.pkl'
roc_data = 'roc_data.csv'
prc_data = 'prc_data.csv'


####################
# ROC and PREREC curves
####################
def plot_rocauc(fpr_rf, tpr_rf, roc_auc):
    plt.figure(1)
    plt.plot(fpr_rf, tpr_rf, label='(AUROC = %0.3f)' % roc_auc)
    plt.title('AUROC' + str(C_))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(pict_name_rocauc)


# Plot precision-recall
def plot_prerec(precision, recall, average_precision):
    plt.figure(2)
    lw = 2
    plt.plot(recall, precision, lw=lw,
             label='Av.PREC = %0.3f' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUCPR' + str(C_))
    plt.legend(loc="upper right")
    plt.savefig(pict_name_prerec)


#########################
# LOAD DATA
#########################
cryptic_training_ = pd.DataFrame()
for chunk in pd.read_csv(cryptic_training, header=0, sep=',', chunksize=1000):
    cryptic_training_ = pd.concat([cryptic_training_, chunk], ignore_index=False)

cryptic_testing_ = pd.DataFrame()
for chunk in pd.read_csv(cryptic_testing, header=0, sep=',', chunksize=1000):
    cryptic_testing_ = pd.concat([cryptic_testing_, chunk], ignore_index=False)

dominant_training_ = pd.DataFrame()
for chunk in pd.read_csv(dominant_training, header=0, sep=',', chunksize=1000):
    dominant_training_ = pd.concat([dominant_training_, chunk], ignore_index=False)

dominant_testing_ = pd.DataFrame()
for chunk in pd.read_csv(dominant_testing, header=0, sep=',', chunksize=1000):
    dominant_testing_ = pd.concat([dominant_testing_, chunk], ignore_index=False)

headers = list(dominant_testing_)


# Generate labels
dominant_training_labels = np.array([1] * len(dominant_training_))
dominant_testing_labels = np.array([1] * len(dominant_testing_))
cryptic_training_labels = np.array([0] * len(cryptic_training_))
cryptic_testing_labels = np.array([0] * len(cryptic_testing_))

dominant_training_labels = dominant_training_labels.reshape(len(dominant_training_labels), 1)
dominant_testing_labels = dominant_testing_labels.reshape(len(dominant_testing_labels), 1)
cryptic_training_labels = cryptic_training_labels.reshape(len(cryptic_training_labels), 1)
cryptic_testing_labels = cryptic_testing_labels.reshape(len(cryptic_testing_labels), 1)

training_labels = np.concatenate((dominant_training_labels, cryptic_training_labels), axis=None)
testing_labels = np.concatenate((dominant_testing_labels, cryptic_testing_labels), axis=None)

# Preprocess data
dominant_training_array = dominant_training_.values
dominant_testing_array = dominant_testing_.values
cryptic_training_array = cryptic_training_.values
cryptic_testing_array = cryptic_testing_.values

joint_training = np.vstack((dominant_training_array, cryptic_training_array))
joint_test = np.vstack((dominant_testing_array, cryptic_testing_array))

scaler = StandardScaler()

scaler.fit(joint_training)
pickle.dump(scaler, open('CrypticScaler.pkl', 'wb'))

scaled_training = scaler.transform(joint_training)
scaled_test = scaler.transform(joint_test)


#########################
# Train the model
#########################
# Define classifier (LR)
lr_model = LogisticRegression(C=C_, penalty="l1", tol=0.01, solver="saga", class_weight='balanced', n_jobs=-1)

# Train classifier
lr_model.fit(scaled_training, training_labels)

# Make predictions
lr_prediction = lr_model.predict_proba(scaled_test)[:, 1]

# Evaluate performance
fpr_lr, tpr_lr, _ = roc_curve(testing_labels, lr_prediction)
roc_auc = auc(fpr_lr, tpr_lr)
plot_rocauc(fpr_lr, tpr_lr, roc_auc)

# Now, calculate and return pre/rec parameters
precision, recall, _ = precision_recall_curve(testing_labels, lr_prediction)
average_precision = average_precision_score(testing_labels, lr_prediction)
plot_prerec(precision, recall, average_precision)

# Get feature imps
lr_importances = lr_model.coef_

importances_ = [float(imp) for imp in lr_importances[0]]
importances_ = np.column_stack((np.asarray(importances_, dtype=np.float32), headers))

with open(importances_output, "w") as f:
    w = csv.writer(f)
    w.writerows(importances_)


############################
# Save
############################
pickle.dump(lr_model, open(model_name, 'wb'))

roc_auc_1 = [roc_auc, 0]
average_precision_1 = [average_precision, 0]


with open(roc_data, "w") as f:
    w = csv.writer(f)
    w.writerow(fpr_lr)
    w.writerow(tpr_lr)
    w.writerow(roc_auc_1)

with open(prc_data, "w") as f:
    w = csv.writer(f)
    w.writerow(precision)
    w.writerow(recall)
    w.writerow(average_precision_1)
