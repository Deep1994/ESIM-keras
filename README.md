# ESIM-keras
This repository is a keras implementation about the ESIM model proposed in paper [Enhanced LSTM for Natural Language Inference](https://arxiv.org/abs/1609.06038). The ESIM model is a strong baseline used for text matching tasks, such as Natural Language Inference(NLI) and paraphrase detection.

# Requirements

+ keras, tested on 2.3.1, with tensorflow 1.14.0 as its backend.
+ python 3.5+
+ NLTK

# Usage 

First, you should download the Quora Question Pairs dataset via [http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv](http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv). We also need GloVe, which is a pretrained word embeddings, you can download it [here](http://nlp.stanford.edu/projects/glove/). Put the glove.840B.300d.txt file to ./.

Atfer that, you can train, test, and save the model via:

```
python run.py
```

The model will be saved to the ./saved_models.

# Notice

Except the GloVe word embeddings, I also use character embeddings obtained from bi-lstm, you can remove it anyway if you do not need it.

# Results
 Model| Dataset | test loss | Accuracy | 
-|-|-|-|
ESIM-Glove | QQP | 0.3737 | 0.8812
