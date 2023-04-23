from datasets import load_dataset
import os
import sys
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)    
    args = parser.parse_args()
    return args

def get_mlm():  
    args = get_args()

    if args.dataset_name == 'bias-bios':
        dataset = load_dataset('parquet', data_files={
            'train' : 'data/bias-bios/biasbios_train.pq', 
            'validation' : 'data/bias-bios/biasbios_val.pq',
            'test' : 'data/bias-bios/biasbios_test.pq'}
            )

        dataset = dataset.remove_columns(["text", "hard_text_untokenized", "text_without_gender"])
        dataset = dataset.rename_column("hard_text", "text")

        dataset['train'].to_json("data/bias-bios/train.json")
        dataset['validation'].to_json("data/bias-bios/validation.json")
        dataset['test'].to_json("data/bias-bios/test.json")

        train_texts  = list(dataset['train']['text'])
        valid_texts = list(dataset['validation']['text'])
        test_texts = list(dataset['test']['text'])


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


    print (f"length : train {len(train_texts)}, val : {len(valid_texts)}, test : {len(test_texts)}")

    data_dir = os.path.join('data', args.dataset_name, 'mlm')
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
        for t in train_texts:
            f.write(t.strip()+'\n')

    with open(os.path.join(data_dir, 'validation.txt'), 'w') as f:
        for t in valid_texts:
            f.write(t.strip()+'\n')

    with open(os.path.join(data_dir, 'test.txt'), 'w') as f:
        for t in test_texts:
            f.write(t.strip()+'\n')

if __name__ == "__main__":
    get_mlm()
    