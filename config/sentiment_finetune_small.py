# i got GPU out of memory thing with "sentiment__finetune.py"


out_dir = "out-sentiment-small"
init_from = "gpt2"
wandb_log = True
wandb_project = "DI725-A1"
wandb_run_name = "finetune_small"

dataset = "customer_service"

# low resourc values:
gradient_accumulation_steps = 2
batch_size = 2           # reduced from 12
block_size = 128         # reduced from 1024
eval_interval = 50
log_interval = 1
eval_iters = 10
eval_only = False
always_save_checkpoint = True

n_layer = 6              # reduced from 12
n_head = 6               # reduced from 12
n_embd = 384             # reduced from 768
dropout = 0.2
bias = True

learning_rate = 1e-4
max_iters = 300
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 50
lr_decay_iters = 300
min_lr = 1e-5

backend = "nccl"
device = "cuda"
dtype = "float16"
compile = False
