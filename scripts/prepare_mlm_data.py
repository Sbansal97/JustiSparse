from datasets import load_dataset
import os
import sys
import pandas as pd
import json
import argparse
from preprocessing import Preprocessing_Tweet

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)    
    args = parser.parse_args()
    return args

def get_mlm():  
    args = get_args()

    if args.dataset_name == 'bias-bios':
        dataset = load_dataset('parquet', data_files={
            'train' : '../../orig-data/bias-bios/biasbios_train.pq', 
            'validation' : '../../orig-data/bias-bios/biasbios_val.pq',
            'test' : '../../orig-data/bias-bios/biasbios_test.pq'}
            )

        dataset = dataset.remove_columns(["text", "hard_text_untokenized", "text_without_gender"])
        dataset = dataset.rename_column("hard_text", "text")

        dataset['train'].to_json("../data/bias-bios/train.json")
        dataset['validation'].to_json("../data/bias-bios/validation.json")
        dataset['test'].to_json("../data/bias-bios/test.json")

        train_texts  = list(dataset['train']['text'])
        valid_texts = list(dataset['validation']['text'])
        test_texts = list(dataset['test']['text'])

    elif args.dataset_name == 'ws':
        for split in ['train', 'dev', 'test']:
            new_data = []
            data = pd.read_csv(f'../data/ws/{split}.tsv' ,sep='\t')
            data['text'] = data['text'].map(Preprocessing_Tweet)

            if split == 'dev':
                split = 'validation'

            for i in range(len(data)) :
                text = data.iloc[i]['text']
                if len(text) > 0:
                    new_data.append({'text': text , 'label' : int(data.iloc[i]['is_hate'])})
            with open(f'../data/ws/{split}.jsonl','w') as fp:
                for item in new_data:
                    fp.write("%s\n" % json.dumps(item))   

    elif args.dataset_name == 'fdcl-new':
        for split in ['trn','dev','tst']:
            new_data = []
            data = pd.read_csv(f'../data/fdcl/toxic_lang_data_pub/ND_founta_{split}_dial_pAPI.csv')
            filtered_data = data[['id','tweet','dialect_argmax','ND_label']]
            filtered_data = filtered_data.rename(columns={"tweet": "text"})
            filtered_data = filtered_data.rename(columns={"dialect_argmax": "t"})
            filtered_data['text'] = filtered_data['text'].map(Preprocessing_Tweet)

            if split=='trn':
                split='train'
            elif split == 'dev':
                split = 'validation'
            elif split == 'tst':
                split = 'test'

            filtered_data.to_json(f'../data/fdcl/{split}.jsonl', orient='records', lines=True)


            # for i in range(len(data)) :
            #     text = data.iloc[i]['text']
            #     if len(text) > 0:
            #         new_data.append({'text': text , 'label' : int(data.iloc[i]['is_hate'])})
                    
    elif args.dataset_name == 'ontonotes':        
        dataset = load_dataset("conll2012_ontonotesv5", "english_v12")
        import pdb
        pdb.set_trace()    
    elif args.dataset_name == 'snli':
        dataset = load_dataset("snli",cache_dir="../data/cache")
        train_texts  = list(dataset['train']['hypothesis'])
        valid_texts = list(dataset['validation']['hypothesis'])
        test_texts = list(dataset['test']['hypothesis'])

        train_texts.extend(list(dataset['train']['premise']))
        valid_texts.extend(list(dataset['validation']['premise']))
        test_texts.extend(list(dataset['test']['premise']))

    else:
        raise NotImplementedError


    # print (f"length : train {len(train_texts)}, val : {len(valid_texts)}, test : {len(test_texts)}")

    # data_dir = os.path.join('data', args.dataset_name, 'mlm')
    # os.makedirs(data_dir, exist_ok=True)

    # with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
    #     for t in train_texts:
    #         f.write(t.strip()+'\n')

    # with open(os.path.join(data_dir, 'validation.txt'), 'w') as f:
    #     for t in valid_texts:
    #         f.write(t.strip()+'\n')

    # with open(os.path.join(data_dir, 'test.txt'), 'w') as f:
    #     for t in test_texts:
    #         f.write(t.strip()+'\n')

if __name__ == "__main__":
    get_mlm()
    