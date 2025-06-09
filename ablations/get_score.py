import json
import os
import argparse
def get_avg(datas,origin):
    total_num = 0
    total_score = 0
    num_0 = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    for i, data in enumerate(datas):
        if origin[i]['document'] == None:
            continue
        total_num += 1
        score = int(data['evaluation'])
        total_score += score
        if score == 0:
            num_0 += 1
        elif score == 1:
            num_1 += 1
        elif score ==2:
            num_2 += 1
        elif score == 3:
            num_3 += 1
    print(f"total suqeustion num: {total_num}")
    print(f"avg_score: {total_score/total_num:.2f}")
    print("Percentage of each score:")
    print(f"num_0: {num_0/total_num*100:.2f}%")
    print(f"num_1: {num_1/total_num*100:.2f}%")
    print(f"num_2: {num_2/total_num*100:.2f}%")
    print(f"num_3: {num_3/total_num*100:.2f}%")

def get_score(datas,origin):   
    prob_num = 0
    total_num = 0
    num_0 = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    cur_0 = None
    cur_1 = None
    cur_2 = None
    cur_3 = None
    for i,data in enumerate(datas):
        if origin[i]['document'] == None:
            continue
        if int(data['problem number']) > prob_num:
            prob_num = int(data['problem number'])
            if cur_0 is not None:
                num_0 += cur_0
            if cur_1 is not None:
                num_1 += cur_1
            if cur_2 is not None:
                num_2 += cur_2
            if cur_3 is not None:
                num_3 += cur_3
            total_num += 1
            cur_0 = None
            cur_1 = None
            cur_2 = None
            cur_3 = None
        if int(data['evaluation']) == 0:
            cur_0 = 1
        elif int(data['evaluation']) == 1:
            cur_1 = 1
        elif int(data['evaluation']) == 2:
            cur_2 = 1
        elif int(data['evaluation']) == 3:
            cur_3 = 1
            
    #print(f"average score: {total_score/total_num:.2f}")
    print(f"total problem num: {total_num}")
    print(f"0이 포함된 문제: {num_0/total_num*100:.2f}%")
    print(f"1이 포함된 문제: {num_1/total_num*100:.2f}%")
    print(f"2이 포함된 문제: {num_2/total_num*100:.2f}%")
    print(f"3이 포함된 문제: {num_3/total_num*100:.2f}%")
    #print(f"num_error: {num_4/total_num*100:.2f}%")

def __main__():
    args = argparse.ArgumentParser()
    args.add_argument("--query", type=str, default="")
    args.add_argument("--mode", type=str, default="")
    args = args.parse_args()

    origin_path = f"/home/minhae/multihop-RAG2/ablation/gpqa/llama/{args.query}.json"
    eval_path = f"/home/minhae/multihop-RAG2/ablation/gpqa/llama/{args.query}_results_{args.mode}.json"
    with open(eval_path, 'r') as file:
        eval = json.load(file)
    with open(origin_path, 'r') as file:
        origin = json.load(file)
    print(f"Running query: {args.query} with {args.mode} mode")
    print("-"*100)
    get_avg(eval,origin)
    print("-"*100)
    get_score(eval,origin)
    print("-"*100)

if __name__ == "__main__":
    __main__()
