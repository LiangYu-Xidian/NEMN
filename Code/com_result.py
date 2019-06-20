def start_load(filename):
	J=1
	result={}
	result_index={}
	for i in open(filename).readlines():
		newi=i.strip().split('\t')
		if J>400:
			return result,result_index
		else:
			result[newi[0]+"\t"+newi[1]]=newi[2]
			result_index[newi[0]+"\t"+newi[1]]=J
		J+=1

def load_drug_drug_list():
	drug_drug_list=[]
	for i in open("combination.txt").readlines():
		newi=i.strip().split('\t')
		drugbankid1=newi[0]
		drugbankid2=newi[1]
		str1=drugbankid1+"\t"+drugbankid2
		drug_drug_list.append(str1)
		drug_drug_list.append(drugbankid2+"\t"+drugbankid1)
	return drug_drug_list

if __name__=="__main__":
	result1,result1_index=start_load("result8.txt")
	#print(str(result1_index))
	result2,result2_index=start_load("result9.txt")
	result3,result3_index=start_load("result10.txt")
	result4,result4_index=start_load("result11.txt")
	result_index={}
	for drug_pair in result1_index.keys():
		if (drug_pair in result2_index.keys()) and (drug_pair in result3_index.keys()) and (drug_pair in result4_index.keys()):
			result_index[drug_pair]=result1_index[drug_pair]+result2_index[drug_pair]+result3_index[drug_pair]+result4_index[drug_pair]
	drug_drug_list=load_drug_drug_list()
	J=0
	fw=open("new_cross_topK","w")
	for com in result_index.keys():
		#print(str(com)+"\t"+str(result1_index.get(com)))
		if com in drug_drug_list:
			fw.write(com+"\t"+"1\t"+str(result_index[com])+"\t"+str(result1_index[com])+"\t"+str(result2_index[com])+"\t"+str(result3_index[com])+"\t"+str(result4_index[com])+"\n")
		else:
			fw.write(com+"\t"+"0\t"+str(result_index[com])+"\t"+str(result1_index[com])+"\t"+str(result2_index[com])+"\t"+str(result3_index[com])+"\t"+str(result4_index[com])+"\n")
fw.flush()
fw.close()
