#coding:utf-8
from gensim.models import Word2Vec
import os

#walks = [map(str, walk) for walk in walks]
def load_walks():
	walks=[]
	for filename in os.listdir("path/result_s_equal"):
		print(filename)
		for i in open("path/result_s_equal//"+str(filename)).readlines():
			newi=i.strip().split('\t')
			walks.append(newi)
	walks = [ walk for walk in walks]
	return walks

walks=load_walks()
#####size 5 dimension window 5 size 
#model = Word2Vec(walks, size=50, window=4, min_count=10, sg=1, workers=8, iter=20, hs=0, negative=100)
model = Word2Vec(walks, size=5,window=4,sg=1, workers=10, iter=40, hs=0, negative=50)
model.wv.save_word2vec_format("embedding/drug_embeding")
