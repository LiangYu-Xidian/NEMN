#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')

def load_auc_data(filename):
	X=[];Y=[]
	for i in open(filename).readlines():
		newi=i.strip().split(',')
		X.append(float(newi[0]))
		Y.append(float(newi[1]))
	X=np.array(X)
	Y=np.array(Y)
	return X,Y


if __name__=="__main__":
	fig = plt.figure()
	#chemical_auc  five_auc  indication_auc  plot_auc.py  side_effect_auc  target_auc  textmining_auc
	all_X,all_Y=load_auc_data("all_auc")
	#atc_X,atc_Y=load_auc_data("atc_auc")
	#textmining_X,textmining_Y=load_auc_data("textmining_auc")
	#chemical_X,chemical_Y=load_auc_data("chemical_auc")
	#indication_X,indication_Y=load_auc_data("indication_auc")
	#side_effect_X,side_effect_Y=load_auc_data("side_effect_auc")
	target_X,target_Y=load_auc_data("equal_auc")
	#lines = plt.plot(all_X, all_Y, atc_X, atc_Y, textmining_X, textmining_Y,chemical_X,chemical_Y,indication_X,indication_Y,side_effect_X,side_effect_Y,target_X,target_Y)
	lines = plt.plot(all_X, all_Y,linestyle=':')
	lines = plt.plot(target_X,target_Y)
	plt.legend(('not_equal','equal'),loc='upper right')
	plt.title('ROC')
	#plt.show()
	plt.savefig('auc_bidui.png')
