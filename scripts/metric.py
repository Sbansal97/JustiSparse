import json
import sys
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# pred1 = [item.strip().split('\t')[1] for item in open(f'models/gab/pfeiffer/cda-before/predictions.txt','r').readlines()][1:]
# pred2 = [item.strip().split('\t')[1] for item in open(f'models/gab/pfeiffer/cda-after/predictions.txt','r').readlines()][1:]

# cnt = 0
# for i,j in zip(pred1, pred2):
#     if i!=j:
#         cnt = cnt + 1
# print (cnt)

# import pdb
# pdb.set_trace()

 
def calc_FPRD_gab(peft = 'pfeiffer', loc = 'none'):
    dataset = 'gab'
    debias = 'cda'

    data_attr = ["lesbian","gay", "bisexual", "transgender", "trans", "queer", "lgbt", "lgbtq", "homosexual", "straight", "heterosexual", "male", "female",
        "nonbinary", "african", "african american", "black", "white", "european", "hispanic", "latino", "latina", "latinx", "mexican", "canadian", "american",
        "asian", "indian", "middle eastern", "chinese", "japanese", "christian", "muslim", "jewish", "buddhist", "catholic", "protestant", "sikh", "taoist", 
        "old", "older", "young", "younger", "teenage", "millenial", "middle aged", "elderly", "blind", "deaf", "paralyzed"]

    pred = [item.strip().split('\t') for item in open(f'models/{dataset}/{peft}/{debias}-{loc}/predictions.txt','r').readlines()][1:]
    df = pd.read_csv('data/iptts77k/test.csv')
    assert len(pred) == len(df)

    overall_fpr = []
    _fpr = defaultdict(lambda : [])
    FPRD = 0

    for i in tqdm(range(len(df))):
        text = df.iloc[i]['Text']
        fpr = int(df.iloc[i]['Label']==0 and int(pred[i][1])==1)
        overall_fpr.append(fpr)
        for attr in data_attr:
            if attr in text:
                _fpr[attr].append(fpr)    

    for k in _fpr:
        _fpr[k] = sum(_fpr[k])/ len(_fpr[k])
    overall_fpr = sum(overall_fpr)/len(overall_fpr)

    for k in _fpr:    
        FPRD = FPRD + abs(_fpr[k] - overall_fpr)        

    print (dataset, peft, debias, loc, " FPRD : ", FPRD)

def calc_FPRD_bios(peft = 'pfeiffer', loc = 'none'):
    dataset = 'bias-bios'
    debias = 'cda'

    _tpr = defaultdict(lambda : {'m' : [], 'f' : []})


    pred = [item.strip().split('\t') for item in open(f'models/{dataset}/{peft}/{debias}-{loc}/predictions.txt','r').readlines()][1:]
    data = [json.loads(item) for item in open('data/bias-bios/test.jsonl','r').readlines()]
    assert len(pred) == len(data)
    for i in tqdm(range(len(data))):
        gender = data[i]['g']
        label = data[i]['p']
        prediction = pred[i][1]
        _tpr[label][gender].append(int(label==prediction))

    RMSE = 0.0
    for k in _tpr:
        RMSE = RMSE + (sum(_tpr[k]['m'])/ len(_tpr[k]['m']) - sum(_tpr[k]['f'])/ len(_tpr[k]['f']))**2
    
    RMSE = (RMSE / len(_tpr))**0.5
    print (dataset, peft, debias, loc, " RMSE : ", RMSE*100)    


def main():
    dataset, peft, loc = sys.argv[1], sys.argv[2], sys.argv[3]
    if dataset=='gab':
        calc_FPRD_gab(peft=peft, loc=loc)
    else:
        calc_FPRD_bios(peft=peft, loc=loc)

main()





