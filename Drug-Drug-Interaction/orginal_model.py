#coding:utf-8
from sklearn import svm
import numpy as np
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.model_selection import cross_val_score,cross_validate
#from sklearn.cross_validation import cross_val_score 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve,auc
from lightgbm.sklearn import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA

def load_vec():
    line_number=0
    node_vec={}
    for i in open("output/embedding").readlines():
        if line_number==0:
            line_number+=1
            continue
        newi=i.strip().split(' ')
        node_vec[newi[0]]=newi[1:]
    return node_vec

def load_mydata():
    node_vec=load_vec()
    fw=open("data_of_drug_drug_interaction","w")
    for i in open("input/CRDInteraction").readlines():
        newi=i.strip().split('\t')
        vec1=node_vec.get(newi[0])
        vec2=node_vec.get(newi[1])
        if newi[2]=="1":
            label="1"
        else:
            label="0"
        if vec1!=None and vec2!=None:
            S=""
            #for vec in vec1:
                #S+=vec+"\t"
            #for vec in vec2:
                #S+=vec+"\t"
            for index_i in range(0,len(vec1)):
                #欧式距离计算方式
                S+=str(pow(float(vec1[index_i])-float(vec2[index_i]),2))+"\t"
            S+=label+"\n"
            fw.write(S)
    fw.flush()
    fw.close()

def load_data():
    X=[]
    Y=[]
    Y1=[]
    for i in open("data_of_drug_drug_interaction").readlines():
        newi=i.strip().split('\t')
        X.append([float(x) for x in newi[:-1]])
        Y1.append(int(newi[-1]))
    return np.array(X),np.array(Y1)

if __name__=="__main__":
    
    load_mydata()
    
    X,Y=load_data()
    print(X.shape,Y.shape)
    
    index = np.random.permutation(X.shape[0])
    X=X[index,:]
    Y=Y[index]
    print(np.sum(Y))
    clf = RandomForestClassifier(n_estimators=200,max_depth=40,min_samples_leaf=20,class_weight="balanced",random_state=0,n_jobs = 8)
    scoring=['roc_auc','recall','f1','average_precision','accuracy']
    scores = cross_validate(clf, X, Y, cv=10,n_jobs=8,scoring=scoring,return_train_score=True)#,scoring='roc_auc'
    auc=scores['test_roc_auc']
    train_auc=scores['train_roc_auc']
    recall=scores['test_recall']
    f1=scores['test_f1']
    aupr=scores['test_average_precision']
    acc=scores['test_accuracy']
    print(str(auc))
    print(str(train_auc))
    print("test_AUC: %0.4f (+/- %0.2f)" % (auc.mean(), auc.std() * 2))
    print("train_AUC: %0.4f (+/- %0.2f)" % (train_auc.mean(), train_auc.std() * 2))
    print("recall: %0.4f (+/- %0.2f)" % (recall.mean(), recall.std() * 2))
    print("f1: %0.4f (+/- %0.2f)" % (f1.mean(), f1.std() * 2))
    print("Aupr: %0.4f (+/- %0.2f)" % (aupr.mean(), aupr.std() * 2))
    print("Accuracy: %0.4f (+/- %0.2f)" % (acc.mean(), acc.std() * 2))
