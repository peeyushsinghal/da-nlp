import os
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def get_subfolders(folder_path):
    subfolders = []
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            subfolders.append(os.path.join(root, dir))
    return subfolders

def get_files(folder_path):
    file_names = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_names.append(os.path.join(root, file))
    return file_names

def get_file_name(full_path):
    return os.path.basename(full_path)

def get_data_df(file_path):
    df = pd.DataFrame(columns =['review','rating'])
    try:
        with open(file_path,'r',encoding = 'utf-8') as f:
            data = f.readlines()

    except:
        print("Error in opening file: {}, empty dataframe returned".format(file_path))
        return df
    

    flag_review = False
    flag_rating = False
    new_rating = 0.0
    new_str =""
    rows_list =[]

    for i in range(len(data)):
        if(data[i]=="<rating>\n"):
                flag_rating = True
                new_rating= "" 
                continue
        if(data[i]=="</rating>\n"):
                flag_rating = False
                continue
        if(flag_rating):
                if(len(data[i])>1):
                    sent = data[i]
                    sent = sent[0:len(sent)-1]
                    if(sent[0]=='\t'):
                        sent = sent[1:len(sent)-1]
                new_rating+=sent

    
        if(data[i]=="<review_text>\n"):
                flag_review = True
                new_str = "" 
                continue
        if(data[i]=="</review_text>\n"):
                flag_review = False
                temp_dict = {'review':new_str,
                             'rating':new_rating}
                rows_list.append(temp_dict)
                continue
        if(flag_review):
                if(len(data[i])>1):
                        sent = data[i]
                        sent = sent[0:len(sent)-1]
                        if(sent[0]=='\t'):
                            sent = sent[1:len(sent)-1]
                new_str+=sent
    return pd.DataFrame(rows_list)

    



def evaluate_roc(probs, y_true):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    
    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def evaluate_confusion_matrix(probs, y_true):
    """
    - Evaluate confusion matrix and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')

    # Get confusion matrix over the test set
    conf_matrix = confusion_matrix(y_true=y_true,y_pred=y_pred)
    print(conf_matrix)
