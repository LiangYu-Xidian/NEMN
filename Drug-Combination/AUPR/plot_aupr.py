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
	all_X,all_Y=load_auc_data("all_aupr")
	atc_X,atc_Y=load_auc_data("atc_aupr")
	textmining_X,textmining_Y=load_auc_data("textmining_aupr")
	chemical_X,chemical_Y=load_auc_data("chemical_aupr")
	indication_X,indication_Y=load_auc_data("indication_aupr")
	side_effect_X,side_effect_Y=load_auc_data("side_effect_aupr")
	target_X,target_Y=load_auc_data("target_aupr")
	lines = plt.plot(all_X, all_Y, atc_X, atc_Y, textmining_X, textmining_Y,chemical_X,chemical_Y,indication_X,indication_Y,side_effect_X,side_effect_Y,target_X,target_Y)

	plt.legend(('six_network', 'atc', 'textmining','chemical','indication','side_effect','target'),loc='upper right')
	plt.title('AUPR')
	#plt.show()
	plt.savefig('aupr.png')
