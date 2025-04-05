# config/sentiment_class.py
# i tried to cheat the don_char config that desgined for one GTX 3070 GPU
# i have gtx 1650 so...
# here we are

compile = False
eval_iters = 20
log_interval = 1
block_size = 256
batch_size = 16
n_layer = 4 # 6 ## readjusted to speed-up training
n_head = 4 # 6
n_embd = 256 #384
dropout = 0.2
max_iters = 1000 ## readjusted to speed-up training
lr_decay_iters = 1000
out_dir = 'out-sentiment-small'
wandb_log = True
wandb_project = 'sentiment-classification'
wandb_run_name = 'sentiment-small-gpt'
