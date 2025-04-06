# config/sentiment_finetune.py

# Output directory for checkpoints
out_dir = "out-sentiment-finetune"

# Initialization method: fine-tuning from pre-trained GPT-2 weights.
init_from = "gpt2"

# WANDB logging settings
wandb_log = True
wandb_project = "DI725-A1"
wandb_run_name = "finetune_sentiment_run"  # You can append a timestamp if desired

# Dataset information
dataset = "customer_service"  # Used to locate meta.pkl, etc.

# Training parameters
gradient_accumulation_steps = 5 * 8  # Adjust to simulate a larger batch size if needed
batch_size = 12
block_size = 1024  # For GPT-2, typically the block size is 1024 tokens
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True

# Model parameters: These should match the pre-trained GPT-2 architecture for compatibility.
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1
bias = True

# Optimizer parameters
learning_rate = 1e-4 #3e-4 if needed
max_iters = 500   
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate scheduler settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 500  # Typically the same as max_iters for fine-tuning
min_lr = 6e-5

# DDP settings
backend = "nccl"

# System settings
device = "cuda"  
dtype = "bfloat16" if device == "cuda" and torch.cuda.is_bf16_supported() else "float16"
compile = False #True
