from vars import *
import os
def filter_by_node():

    PATH_F=os.path.join(PATH_D,"targets_t5small_egv_beam.txt")
  
    with open(PATH_F, "r") as f:
        lines=f.readlines()
  #  count_nodes(lines)
    n_nodes=[]
    for line in lines:
        #square_occ=count_square(line)
        line=line.replace("]]]]]]","]").replace("]]]]]","]").replace("]]]]","]").replace("]]]","]").replace("]]","]")
        nodes=line.split("]")
        n_nodes.append(len(nodes) - 1) #cut \n character
    #print(len(lines[10].replace("[[","[").replace("]]]]]]","]").replace("]]]]]","]").replace("]]]]","]").replace("]]]","]").replace("]]","]").split("]")) - 1)
    PATH_O=os.path.join(PATH_D,"ts_num_nodes_for_line.txt")
    with open(PATH_O,"a")as f:
        for node in n_nodes:
            f.write(str(node) + '\n')
        

    # print(len(f_list))
def count_square(line):
    from itertools import groupby
    groups = groupby(line)
    result = [(label, sum(1 for _ in group)) for label, group in groups]
    for r in result:
        if(r[0] == ']'):
            if max < r[1]:
                max=r[1]
    return max

def score_f_nodes(num_nodes=[1,2,3,4,5,6,7],m_type=""):
    PATH_F=os.path.join(PATH_D,"num_nodes_for_line.txt")
    #METEOR
    aggregator=[]
    with open("meteor/barte.txt","r") as f:
        meteor=f.readlines()
    with open("rouge/barte1.txt","r") as f:
        rouge1=f.readlines()
    with open("rouge/barte2.txt","r") as f:
        rouge2=f.readlines()
    with open("rouge/barteL.txt","r") as f:
        rougeL=f.readlines()
    with open("bleu/results_barte.txt","r") as f:
        bleu=f.readlines()
    # with open("rouge/results-L.txt","r") as f:
    #     rougeL=f.readlines()
    with open("bleurt/bb.txt","r") as f:
        bleurt=f.readlines()
    with open("BARTScore/bart.txt","r") as f:
        bartscore=f.readlines()
    # with open("nubia/scores.txt","r") as f:
    #     nub=f.readlines()
    with open("bert_score/barte.txt","r") as f:
        bert_s=f.readlines()
            
    # with open("repts.txt","r") as f:
    #     repts=f.readlines()
    with open(PATH_F,"r") as f:
        lines=f.readlines()
    # with open("questeval/results.txt","r") as f:
    #     qeval_l=f.readlines()
    with open("questeval/bart.txt","r") as f:
         qeval_l=f.readlines()
    ind=0
    if m_type=="bart":
        print("List size: {}".format(len(lines)))
        for nn in num_nodes:
            ex_index=[]
            ind=0
            for i in range(0,len(lines),4):
                if int(lines[i])  == nn  and nn < 7:
                    ex_index.append(ind)
                else:
                    if nn >= 7:
                        if int(lines[i]) >= nn:
                            ex_index.append(ind)
                ind+=1
            # print(calc_score("QuestEval",ex_index,nn,qeval_l))
            # print(calc_score("Bleu",ex_index,nn,bleu))
            # print(calc_score("Bleurt",ex_index,nn,bleurt))
            #print(calc_score("BARTSore",ex_index,nn,bartscore))
            print(calc_score("Meteor",ex_index,nn,meteor))
            print(calc_score("Rouge1",ex_index,nn,rouge1))
            print(calc_score("Rouge2",ex_index,nn,rouge2))
            print(calc_score("RougeL",ex_index,nn,rougeL))
      
    else:
        for nn in num_nodes:
            ex_index=[]
            ind=0
            for line in lines:
                if int(line)  == nn  and nn < 7:
                    ex_index.append(ind)
                else:
                    if nn >= 7:
                        if int(line) >= nn:
                            ex_index.append(ind)
                ind+=1
    #print("List size: {}".format(len(ex_index)))  
            print(calc_score("Meteor",ex_index,nn,meteor))
            print(calc_score("Rouge1",ex_index,nn,rouge1))
            print(calc_score("rouge2",ex_index,nn,rouge2))
            print(calc_score("rougeL",ex_index,nn,rougeL))
            #print(calc_score("Bleurt",ex_index,nn,bleurt))
            #print(len(nub))
            #print(calc_score("Nubia",ex_index,nn,nub))
            print(calc_score("Bert-score",ex_index,nn,bert_s))
            print(calc_score("Bleu",ex_index,nn,bleu))
            # print(calc_score("RR",ex_index,nn,repts))
    #print(calc_score("QuestEval",ex_index,nn,qeval_l))

    
  


def calc_score(name,ex_index,num_nodes,lines=[]):
    import numpy as np
    scores=[]
    print(len(ex_index))
    for c in ex_index:
        scores.append(float(lines[c]))
    return {name+"-" + str(num_nodes) : np.mean(scores)}
