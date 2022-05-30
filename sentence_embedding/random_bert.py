import argparse
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
import os
import pickle


def run(file, tokenizer, model):

    with open(args.source_text+'/'+file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]


    b = []
    print("Starts to create layers")
    for line in lines:
        encoded_input = tokenizer(line.strip(), return_tensors='pt')
        outputs = model(**encoded_input)
        encoded_layers = outputs[0]
        token_vecs = encoded_layers[0]
        sentence_embedding = token_vecs[0]

        b.append(sentence_embedding.detach().numpy())

        path = './'+args.model+'/layer '+str(12)
        if not os.path.exists(path):
            os.makedirs(path)
        newfile = path +'/'+file.split('.')[0]+'_'+args.model+'_embedding.dat'
        with open(newfile, 'wb') as f:
            pickle.dump(b, f)
        print("Layer 12 is done!")


def main(args):
    configuration = BertConfig.from_pretrained('sentence_embedding/config.json')
    model = BertModel(configuration)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    model.eval()
    files = os.listdir(args.source_text)
    for file in files:
        run(file, tokenizer, model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert_random')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--source_text', type=str, default='./sentences')    #FOLDER!!

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(args)
    main(args)
