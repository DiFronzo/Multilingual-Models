import argparse
import torch
from transformers import BertTokenizer, BertModel, XLMRobertaModel, XLMRobertaTokenizer
import os
import pickle


def run(file, tokenizer, model):
    with open(args.source_text + '/' + file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

    print("Starts to create layers for " + file.split('.')[0])
    for layer in range(1, 13):
        b = []
        for line in lines:
            if args.model == 'bert':
                sent = "[CLS] " + line.strip() + " [SEP]"
            else:
                sent = line.strip()
            tokenized_text = tokenizer.tokenize(sent)
            if len(tokenized_text) > args.max_len:
                tokenized_text = tokenized_text[:args.max_len]
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)

            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            with torch.no_grad():
                outputs = model(tokens_tensor, segments_tensors, output_hidden_states=True)
                encoded_layers = outputs[-1][layer]
                token_vecs = encoded_layers[0]
                if args.model == 'bert':
                    sentence_embedding = token_vecs[0]
                else:
                    sentence_embedding = torch.mean(token_vecs, dim=0)
                b.append(sentence_embedding.detach().numpy())

        path = './' + args.model + '/layer ' + str(layer)
        if not os.path.exists(path):
            os.makedirs(path)
        newfile = path + '/' + file.split('.')[0] + '_' + args.model + '_embedding.dat'
        with open(newfile, 'wb') as f:
            pickle.dump(b, f)

        print("Layer " + str(layer) + " is done!")



def main(args):
    if args.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = BertModel.from_pretrained("bert-base-multilingual-cased")
    elif args.model == 'xlm-R':
        tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
    model.eval()
    files = os.listdir(args.source_text)
    for file in files:
        run(file, tokenizer, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert', choices=['bert', 'xlm-R'])
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--source_text', type=str, default='./sentences') #FOLDER!!

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(args)
    main(args)
