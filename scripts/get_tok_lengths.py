import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer


tok = AutoTokenizer.from_pretrained('bert-base-uncased')
data = [line.strip() for line in open('/usr1/datasets/sumita/comp-ethics/project/JustiSparse/corpora/cda/bias_bios_train.txt')]

tok_data = [tok.tokenize(i) for i in tqdm(data)]
len_tok_data = [len(i) for i in tok_data]

print('95th percentile ', np.percentile(len_tok_data, 95))
print('95th percentile ', np.percentile(len_tok_data, 99))