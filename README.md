# DI725 Assignment 1: Sentiment Classification with nanoGPT

(I over emphasized the comments by spamming "#" where i changed in the codes, you can search more than one "#" for the changes
##########for example)

This repository contains my implementation for Assignment 1 of the DI725 course. The goal was to adapt the [nanoGPT](https://github.com/karpathy/nanoGPT) codebase for sentiment classification of customer service conversations. The model was trained using both a from-scratch approach and fine-tuning of a pre-trained GPT-2.

Preprocessing

PRE- preprocessing and vocabulary creation are handled inside the notebook:
```EDA.ipynb```
This notebook:
Filters columns to keep only customer_sentiment and conversation
Applies lowercasing, punctuation removal, and whitespace normalization
Performs a stratified train/validation split
and it creates:
`cleaned_train.csv`
`val_split.csv`
`meta.pkl`
All saved under `data/customer_service/.`

The `meta.pkl` file is later used by the `SentimentDataset` class (in `sentiment_dataset.py`) to convert each cleaned conversation into a sequence of token IDs during training. It also maps sentiment labels into numeric classes. 

Training:

Train from Scratch: 
``` python train.py config/sentiment_class.py ```

(can be inferenced with ```sentiment-sample.py``` )

Fine-Tune Pretrained GPT-2: ```python train.py config/sentiment_finetune_small.py```

If you encounter memory issues, use the small version or reduce model size in the config.

Evaluation:

Use the provided notebook to evaluate the model:

jupyter notebook ```evaluation_notebook.ipynb```

The notebook:

Loads the model from `ckpt.pt`

Runs predictions on the validation set

Computes accuracy, F1-score, and displays the confusion matrix

Logging:

Training progress and metrics are tracked using Weights & Biases (wandb.ai). Make sure that config includes the correct WandB project name and is enabled.

Acknowledgements:

Code adapted from nanoGPT (https://github.com/karpathy/nanoGPT)

Assignment starter template from @caglarmert (https://github.com/caglarmert) 

sorry for the bad explanations
the traning (1000 iterations on gtx1650 🙃)
(https://wandb.ai/delfin-b-metu-middle-east-technical-university/(1000)sentiment-classification/runs/4peo5b0i?nw=nwuserdelfinb)
the fine-tuning_small:
(https://wandb.ai/delfin-b-metu-middle-east-technical-university/DI725-A1/runs/9qaa6uo0?nw=nwuserdelfinb)
