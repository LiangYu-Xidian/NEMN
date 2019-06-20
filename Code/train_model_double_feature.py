#coding:utf-8
import random
from sklearn import svm
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn.cross_validation import cross_val_score 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,precision_recall_curve,average_precision_score
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.neighbors import KNeighborsClassifier

plt.switch_backend('agg')

def load_vec():
	node_vec={}
	J=0
	for i in open("embedding/drug_embeding").readlines():
		if J==0:
			J+=1
			continue
		newi=i.strip().split(' ')
		node_vec[newi[0]]=newi[1:]
	return node_vec

def load_node():
	bankid_node={}
	node_bankid={}
	for i in open("node/Drug_Node_n").readlines():
		newi=i.strip().split('\t')
		bankid_node[newi[0]]=newi[1]
		node_bankid[newi[1]]=newi[0]
	return bankid_node,node_bankid

def load_train(bankid_node,node_vec):
	#positive
	positive_pair=[]
	train=[]
	for i in open("drug-combination/train/DrugBank_DrugBank_Approved").readlines():
		newi=i.strip().split('\t')
		if (newi[0] in bankid_node.keys()) and (newi[1] in bankid_node.keys()):
			bankid1=bankid_node.get(newi[0])
			bankid2=bankid_node.get(newi[1])
			positive_pair.append(str(bankid1)+" "+str(bankid2))
			vec1=node_vec.get(bankid1)
			vec2=node_vec.get(bankid2)
			vec=[]
			vec.append(1)
			#for index_i in range(0,len(vec1)):
			#	vec.append(pow(float(vec1[index_i])-float(vec2[index_i]),2))
			for v1 in vec1:
				vec.append(v1)
			for v2 in vec2:
				vec.append(v2)
			train.append(vec)
			vec=[]
			vec.append(1)
			for v2 in vec2:
				vec.append(v2)
			for v1 in vec1:
				vec.append(v1)
			train.append(vec)
	#negative   random
	train=np.array(train)
	return train,positive_pair

def load_cross(bankid_node,node_vec,positive_pair):
	cross=[]
	for i in open("drug-combination/cross/drugbank_drugbank_1284_feature_cv").readlines():
		newi=i.strip().split('\t')
		if (newi[0] in bankid_node.keys()) and (newi[1] in bankid_node.keys()):
			bankid1=bankid_node.get(newi[0])
			bankid2=bankid_node.get(newi[1])
			#print(bankid1,bankid2)
			positive_pair.append(str(bankid1)+" "+str(bankid2))
			vec1=node_vec.get(bankid1)
			vec2=node_vec.get(bankid2)
			vec=[]
			vec.append(1)
			#for index_i in range(0,len(vec1)):
			#	vec.append(pow(float(vec1[index_i])-float(vec2[index_i]),2))
			for v1 in vec1:
				vec.append(v1)
			for v2 in vec2:
				vec.append(v2)
			cross.append(vec)
			vec=[]
			vec.append(1)
			for v2 in vec2:
				vec.append(v2)
			for v1 in vec1:
				vec.append(v1)
			cross.append(vec)
	#cross=np.array(cross)
	negative=[]
	for time in range(800):
		bankid1=random.randint(0,1283)
		bankid2=random.randint(0,1283)
		if (bankid1!=bankid2) and (str(bankid1)+" "+str(bankid2) not in positive_pair) and (str(bankid2)+" "+str(bankid1) not in positive_pair):
			positive_pair.append(str(bankid1)+" "+str(bankid2))
			vec1=node_vec.get(str(bankid1))
			#print bankid1,vec1
			vec2=node_vec.get(str(bankid2))
			vec=[]
			vec.append(0)
			if(vec1!=None and vec2!=None):
				#for index_i in range(0,len(vec1)):
				#	vec.append(pow(float(vec1[index_i])-float(vec2[index_i]),2))
				for v1 in vec1:
					vec.append(v1)
				for v2 in vec2:
					vec.append(v2)
				cross.append(vec)
				vec=[]
				vec.append(0)
				for v2 in vec2:
					vec.append(v2)
				for v1 in vec1:
					vec.append(v1)
				cross.append(vec)
	cross=np.array(cross)
	return cross

if __name__=="__main__":
	node_vec=load_vec()
	bankid_node,node_bankid=load_node()
	traindata,positive_pair=load_train(bankid_node,node_vec)
	crossdata=load_cross(bankid_node,node_vec,positive_pair)
	print(traindata.shape)
	print(crossdata.shape)
	traindata=np.concatenate((traindata,crossdata),axis=0)
	fw2=open("drug_drug_interaction","w")
	for i in range(0,traindata.shape[0]):
		S=""
		for j in range(0,traindata.shape[1]):
			S+=str(traindata[i][j])+"\t"
		S=S[:-1]+"\n"
		fw2.write(S)
	fw2.flush()
	fw2.close()
	X=traindata[:,1:]
	Y=traindata[:,0]
	print(X.shape,Y.shape)
	print(Y)
	
	newY=[]
	for y in Y:
		#print y
		newY.append(round(float(y)))
	Y=np.array(newY)
	
	index = np.random.permutation(X.shape[0])
	X=X[index,:].astype('float64')
	Y=Y[index].astype('float64')
	print(X.shape,Y.shape)
	print("train_model!")
	clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2,class_weight="balanced",n_jobs = 10)
	#clf=LogisticRegression(class_weight="balanced",max_iter=500)
	#clf = SVC(probability=True,class_weight="balanced")
	#clf=KNeighborsClassifier(n_neighbors=5)
	
	scoring=['roc_auc','recall','f1','average_precision','accuracy']
	scores = cross_validate(clf, X, Y, cv=10,n_jobs=10,scoring=scoring,return_train_score=True)#,scoring='roc_auc'
	auc_v=scores['test_roc_auc']
	train_auc=scores['train_roc_auc']
	recall=scores['test_recall']
	f1=scores['test_f1']
	aupr=scores['test_average_precision']
	acc=scores['test_accuracy']
	print(str(auc))
	print(str(train_auc))
	print("test_AUC: %0.4f (+/- %0.2f)" % (auc_v.mean(), auc_v.std() * 2))
	#print("train_AUC: %0.4f (+/- %0.2f)" % (train_auc.mean(), train_auc.std() * 2))
	print("recall: %0.4f (+/- %0.2f)" % (recall.mean(), recall.std() * 2))
	print("f1: %0.4f (+/- %0.2f)" % (f1.mean(), f1.std() * 2))
	print("Aupr: %0.4f (+/- %0.2f)" % (aupr.mean(), aupr.std() * 2))
	print("Accuracy: %0.4f (+/- %0.2f)" % (acc.mean(), acc.std() * 2))
	
