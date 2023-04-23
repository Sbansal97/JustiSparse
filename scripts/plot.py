from matplotlib import pyplot as plt
from collections import defaultdict
import pandas as pd

with open('peft_cda_bios_bias_mlm.txt', 'r') as f:
    data = f.readlines()

for i in range(len(data)):
    data[i] = data[i].strip().split('\t')
    data[i][1] = float(data[i][1].split('checkpoint-')[1].split('/')[0])
    data[i][2] = float(data[i][2])

df = pd.DataFrame(data, columns=["method", "ckpt", "bias"])

_dic = defaultdict(lambda : [])

for i in range(len(df)):
    _dic[df.iloc[i,0]].append((df.iloc[i,1],df.iloc[i,2]))

plt.figure(figsize=(12,8))

for k in _dic:
    _dic[k] = sorted(list(set(_dic[k])), key = lambda x : x[0])
    x = [a for a,b in _dic[k]]
    y = [b for a,b in _dic[k]]
    plt.plot(x,y,label=k)
    plt.scatter(x,y)

plt.legend()
plt.xlabel('Training steps')
plt.ylabel('crow-s (lower is better)')
plt.title('bias-bios + CDA + mlm')


plt.savefig('plot.png')

    

# import pdb
# pdb.set_trace()
