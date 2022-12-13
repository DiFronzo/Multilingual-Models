# Multilingual-Models
Pre-trained multilingual models to encode language identity for the Scandinavian languages.

## Installation

Recommend **Python 3.8** or higher, **[PyTorch 1.6.0](https://pytorch.org/get-started/locally/)** or higher and **[transformers v4.6.0](https://github.com/huggingface/transformers)** or higher. The code does **not** work with Python 2.7.

## Usage
```
$ pip3 install requirements.txt
```

Create data for mBERT and/or XLM-R:
```
$ python3 sentence_embedding/main.py
```
Parameters allowed:
* '--model' (bert (def.) or xlm-R)
* '--max_len' (default is 512)
* '--source_text' (default is './sentences')

```
$ python3 typological_features/main.py
```
Parameters allowed:
* '--model' (bert (def.) or xlm-R)
* '--num_train_epochs' (default is 5)
* '--hidden_dropout_prob' (default is 0.5)
* '--learning_rate' (default is 1e-2)
* '--input_size' (768 (def.) or 1024)
* '--hidden_dim' (default is 100)
* '--train_batch_size' (default is 512)
* '--layer' (default is 1)

NB! the data in the folder "sentences" is not under a MIT license, Swedish: © 1998 - 2022 Deutscher Wortschatz. All rights reserved., Norwegian: © Store Norske Leksikon, Danish: © Den Store Danske.

## Result

As shown in the table below, mBERT outperforms XLM-R possibly because the Next Sentence Prediction (NSP) task plays an important role in keeping language identity when training the BERT model. BERT needs to identify whether two sentences are from the same language and if they are adjacent. Both mBERT and XLM-R are trained on distinct lexicon text. The baseline is Random BERT, that is more of a real-world experience for detecting languages. Random BERT randomizes all weights of each layer at the lexical layer (layer 0) as it is done in the paper by Tenney et al [[1](https://arxiv.org/abs/1905.06316)].

| Lang | mBERT | XLM-R | Baseline* |
|-------------------------------------|-------------------------------------|-------------------------------------|----------------------------------------|
| nb                                   | 93.45\%                             | 89.07\%                             | 65.05\%                                |
| da                                  | 95.56\%                             | 91.81\%                             | 64.76\%                                |
| sv                                  | 89.14\%                             | 81.64\%                             | 48.47\%                                |
| en                                  | 82.45\%                             | 82.01\%                             | N/A                                    

*Baseline is Random BERT.
