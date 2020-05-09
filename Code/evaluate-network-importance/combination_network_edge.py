#coding:utf-8
import numpy as np
import random
from scipy import stats
import matplotlib as pl
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #Specify the default font 
mpl.rcParams['axes.unicode_minus'] = False # Resolve save image is a minus sign '-' displayed as a box
sns.set_context("talk")

################################
#plot picture
def plot_picture(x1,x2):
	#bw=0.02,bw=0.02,
	sns.kdeplot(x1,label=u'The weight distribution of the corresponding edge of the random drug pair')
	sns.kdeplot(x2,label=u'The weight distribution of the edge corresponding to the drug combination')
	x1_mean=np.mean(np.array(x1))
	x2_mean=np.mean(np.array(x2))
	x1_list=[]
	x2_list=[]
	for i in range(0,100):
		x1_list.append(x1_mean)
		x2_list.append(x2_mean)
	
	y= np.linspace(0, 6, 100)
	print y.shape
	plt.title(u'Weight distribution of edges of drug combinations vs. weight distribution of edges of random drug pairs')
	#plt.xlim(xmin=0)
	plt.xlabel(u'The weight of edge')
	plt.ylabel(u'frequency')
	plt.plot(x1_list,y, linewidth=2.5,label=u'The weighted mean of the side opposite the random drug')
	plt.plot(x2_list,y, linewidth=2.5,label=u'The weighted mean of the edges of the drug combinations')
	#sns.distplot(x)
	plt.legend()
	plt.show()
###############################
def load_drugbankid_id():
	#get drug node map int
	drugbankid_id={}
	for i in open("Drug_Node").readlines():
		newi=i.strip().split('\t')
		drugbankid_id[newi[0]]=int(newi[1])
	return drugbankid_id

def load_data(filename,drugbankid_id):
	drug_node_number=len(drugbankid_id.keys())
	drug_network=np.zeros((drug_node_number,drug_node_number))
	#drug_network=drug_network+100
	Edge=[]
	for i in open(filename).readlines():
		newi=i.strip().split('\t')
		if newi[0] in drugbankid_id.keys() and newi[1] in drugbankid_id.keys():
			drugbankid1=drugbankid_id.get(newi[0])
			drugbankid2=drugbankid_id.get(newi[1])
			edge=newi[2]
			drug_network[drugbankid1][drugbankid2]=edge
	druglist=drugbankid_id.keys()
	for i in range(0,len(druglist)-1):
		for j in range(i+1,len(druglist)-1):
			Edge.append(drug_network[drugbankid_id.get(druglist[i])][drugbankid_id.get(druglist[j])])
	return drug_network,Edge

def load2_data(filename,drugbankid_id):
	drug_node_number=len(drugbankid_id.keys())
	print drug_node_number
	drug_network=np.zeros((drug_node_number,drug_node_number))
	print type(drug_network)
	print drug_network.shape
	drug_network=drug_network+100
	print drug_network.shape
	Edge=[]
	for i in open(filename).readlines():
		newi=i.strip().split('\t')
		edge=float(newi[2])
		drugbankid1=drugbankid_id.get(newi[0])
		drugbankid2=drugbankid_id.get(newi[1])
		if drugbankid1==None or drugbankid2==None:
			continue
		#print drugbankid1,drugbankid2
		#print drug_network.shape
		drug_network[drugbankid1][drugbankid2]=edge
		Edge.append(float(edge))
	druglist=drugbankid_id.keys()
	'''
	for i in range(0,len(druglist)-1):
		for j in range(i+1,len(druglist)-1):
			Edge.append(drug_network[drugbankid_id.get(druglist[i])][drugbankid_id.get(druglist[j])])
	'''
	return drug_network,Edge

def random_edge_weight(Edge,number=100000):
	''''''
	length=len(Edge)-1
	index=[]
	for i in xrange(1000000):
		index.append(float(Edge[random.randint(0,length)]))
	return index

def load_combination_edge(drugbankid_id,network):
	combination_edge=[]
	mySum=[]
	for i in open("DrugBank_DrugBank_Approved").readlines():
		newi=i.strip().split('\t')
		drug1=newi[0]
		drug2=newi[1]
		x=drugbankid_id.get(drug1)
		y=drugbankid_id.get(drug2)
		if x==None or y==None:
			continue
		#print x,y,drug1,drug2
		if float(network[x][y])==100.0 :
			continue
		combination_edge.append(drug1+"\t"+drug2+"\t"+str(network[x][y]))
		mySum.append(float(network[x][y]))
	return combination_edge,mySum


def mywrite(result,filename="network1"):
	fwriter=open(filename,"w")
	for res in result:
		fwriter.write(res+"\n")
	fwriter.flush()
	fwriter.close()

#
def Analysize_sp(weight):
	mymap={}
	for i in weight:
		if i in mymap.keys():
			mymap[i]+=1
		else:
			mymap[i]=1
	print mymap

if __name__=="__main__":
	drugbankid_id=load_drugbankid_id()         #drugbankid index
	print len(drugbankid_id.keys())
	#network,Edge=load_data("drug_drug_category_remove_0",drugbankid_id)          #drug matrix
	#network,Edge=load_data("chemial_chemial_textmining",drugbankid_id)
	network,Edge=load2_data("drug_drug_ATC",drugbankid_id)
	#network,Edge=load_data("drug_drug_side_effect",drugbankid_id)
	print len(Edge)
	random_weight=random_edge_weight(Edge)  #remove edge matrix
	print len(random_weight)
	#print random_weight
	combination_edge,mySum=load_combination_edge(drugbankid_id,network)
	print len(mySum)
	print np.mean(np.array(random_weight)),np.mean(np.array(mySum))
	s,p=stats.ranksums(random_weight,mySum)
	print s,p
	print mySum
	plot_picture(random_weight,mySum)
	#Analysize_sp(random_weight)
	#Analysize_sp(mySum)
	#plot_picture(mySum)
	mywrite(combination_edge,"drug_target_distance_combintion")
