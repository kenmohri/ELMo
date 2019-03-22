Pre-trained ELMo Representations for Many Languages
===================================================

詳細は [Pre-trained ELMo Representations for Many Languages](https://github.com/HIT-SCIR/ELMoForManyLangs) 参照。本レポジトリは、Pre-requirements にしめす環境で動作するように biLM.py に変更を加えたものである。

## Pre-requirements

* **must** python >= 3.6 (if you use python3.5, you will encounter this issue https://github.com/HIT-SCIR/ELMoForManyLangs/issues/8)
* CUDA 10
* pytorch 1.0
* other requirements from allennlp

### Install the package

You need to install the package to use the embeddings with the following commends
```
python setup.py install
```

### Set up the `config_path`
After unzip the model, you will find a JSON file `${lang}.model/config.json`.
Please change the `"config_path"` field to the relative path to 
the model configuration `cnn_50_100_512_4096_sample.json`.
For example, if your ELMo model is `zht.model/config.json` and your model configuration
is `zht.model/cnn_50_100_512_4096_sample.json`, you need to change `"config_path"`
in `zht.model/config.json` to `cnn_50_100_512_4096_sample.json`.

If there is no configuration `cnn_50_100_512_4096_sample.json` under `${lang}.model`,
you can copy the `configs/cnn_50_100_512_4096_sample.json` into `${lang}.model`,
or change the `"config_path"` into  `configs/cnn_50_100_512_4096_sample.json`.

See [issue 27](https://github.com/HIT-SCIR/ELMoForManyLangs/issues/27) for more details. 


### Use ELMoForManyLangs in command line

Prepare your input file in the [conllu format](http://universaldependencies.org/format.html), like
```
1   Sue    Sue    _   _   _   _   _   _   _
2   likes  like   _   _   _   _   _   _   _
3   coffee coffee _   _   _   _   _   _   _
4   and    and    _   _   _   _   _   _   _
5   Bill   Bill   _   _   _   _   _   _   _
6   tea    tea    _   _   _   _   _   _   _
```
Fileds should be separated by `'\t'`. We only use the second column and space (`' '`) is supported in
this field (for Vietnamese, a word can contains spaces).
Do remember tokenization!

When it's all set, run

```
$ python -m elmoformanylangs test \
    --input_format conll \
    --input /path/to/your/input \
    --model /path/to/your/model \
    --output_prefix /path/to/your/output \
    --output_format hdf5 \
    --output_layer -1
```

It will dump an hdf5 encoded `dict` onto the disk, where the key is `'\t'` separated
words in the sentence and the value is it's 3-layer averaged ELMo representation.
You can also dump the cnn encoded word with `--output_layer 0`,
the first layer of the LsTM with `--output_layer 1` and the second layer
of the LSTM with `--output_layer 2`.  
We are actively changing the interface to make it more adapted to the 
AllenNLP ELMo and more programmatically friendly.

### Use ELMoForManyLangs programmatically

Thanks @voidism for contributing the API.
By using `Embedder` python object, you can use ELMo into your own code like this:

```python
from elmoformanylangs import Embedder

e = Embedder('/path/to/your/model/')

sents = [['今', '天', '天氣', '真', '好', '阿'],
['潮水', '退', '了', '就', '知道', '誰', '沒', '穿', '褲子']]
# the list of lists which store the sentences 
# after segment if necessary.

e.sents2elmo(sents)
# will return a list of numpy arrays 
# each with the shape=(seq_len, embedding_size)
```

#### the parameters to init Embedder:
```python
class Embedder(model_dir='/path/to/your/model/', batch_size=64):
```
- **model_dir**: the absolute path from the repo top dir to you model dir.
- **batch_size**: the batch_size you want when the model inference, you can specify it properly according to your gpu/cpu ram size. (default: 64)

#### the parameters of the function sents2elmo:
```python
def sents2elmo(sents, output_layer=-1):
```
- **sents**: the list of lists which store the sentences after segment if necessary.
- **output_layer**: the target layer to output. 
    -  0 for the word encoder
    -  1 for the first LSTM hidden layer
    -  2 for the second LSTM hidden layer
    -  -1 for an average of 3 layers. (default)
    -  -2 for all 3 layers

## Training Your Own ELMo

Please run 
```
$ python -m elmoformanylangs.biLM train -h
```
to get more details about the ELMo training. 

Here is an example for training English ELMo.
```
$ less data/en.raw
... (snip) ...
Notable alumni
Aris Kalafatis ( Acting )
Labour Party
They build an open nest in a tree hole , or man - made nest - boxes .
Legacy
... (snip) ...

$ python -m elmoformanylangs.biLM train \
    --train_path data/en.raw \
    --config_path configs/cnn_50_100_512_4096_sample.json \
    --model output/en \
    --optimizer adam \
    --lr 0.001 \
    --lr_decay 0.8 \
    --max_epoch 10 \
    --max_sent_len 20 \
    --max_vocab_size 150000 \
    --min_count 3
```
However, we
need to add that the training process is not very stable.
In some cases, we end up with a loss of `nan`. We are actively working on that and hopefully
improve it in the future.

## Citation

If our ELMo gave you nice improvements, please cite us.

```
@InProceedings{che-EtAl:2018:K18-2,
  author    = {Che, Wanxiang  and  Liu, Yijia  and  Wang, Yuxuan  and  Zheng, Bo  and  Liu, Ting},
  title     = {Towards Better {UD} Parsing: Deep Contextualized Word Embeddings, Ensemble, and Treebank Concatenation},
  booktitle = {Proceedings of the {CoNLL} 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies},
  month     = {October},
  year      = {2018},
  address   = {Brussels, Belgium},
  publisher = {Association for Computational Linguistics},
  pages     = {55--64},
  url       = {http://www.aclweb.org/anthology/K18-2005}
}
```

Please also cite the 
[NLPL Vectors Repository](http://wiki.nlpl.eu/index.php/Vectors/home)
for hosting the models.
```
@InProceedings{fares-EtAl:2017:NoDaLiDa,
  author    = {Fares, Murhaf  and  Kutuzov, Andrey  and  Oepen, Stephan  and  Velldal, Erik},
  title     = {Word vectors, reuse, and replicability: Towards a community repository of large-text resources},
  booktitle = {Proceedings of the 21st Nordic Conference on Computational Linguistics},
  month     = {May},
  year      = {2017},
  address   = {Gothenburg, Sweden},
  publisher = {Association for Computational Linguistics},
  pages     = {271--276},
  url       = {http://www.aclweb.org/anthology/W17-0237}
}
```