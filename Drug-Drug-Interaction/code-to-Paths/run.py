#coding:utf-8
import os
for line in open("conf").readlines():
    newline=line.strip().split(':')
    print(newline[0])
    myin = "../input/networks/"+str(newline[0])
    print(myin)
    myoutput="../output/paths/"+newline[0]+"_seq"
    print(myoutput)
    print("python main.py --input "+myin+" --output "+myoutput+" --num-walks "+ str(newline[1]))
    b=os.popen("python main.py --input "+myin+" --output "+myoutput+" --num-walks "+ str(newline[1]))
    print(b)
