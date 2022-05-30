# Multilingual-Models
Project Work in IMT4308

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