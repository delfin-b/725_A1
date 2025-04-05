import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import GPT, GPTConfig  # modified GPT model 
from sentiment_dataset import SentimentDataset  #  custom dataset class
import pickle
import wandb
from sklearn.metrics import f1_score  


wandb.init(project="DI725_Sentiment_Analysis", name="sentiment_run", config={
    "batch_size": 32,
    "max_epochs": 20,
    "learning_rate": 3e-4,
    "block_size": 256,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "dropout": 0.1,
    "bias": True,
    "num_classes": 3
})

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters from wandb config
batch_size = wandb.config.batch_size
max_epochs = wandb.config.max_epochs
learning_rate = wandb.config.learning_rate
block_size = wandb.config.block_size
n_layer = wandb.config.n_layer
n_head = wandb.config.n_head
n_embd = wandb.config.n_embd
dropout = wandb.config.dropout
bias = wandb.config.bias

# Load meta to get vocabulary size
with open('data/customer_service/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']

# Create model configuration and initialize model for sentiment analysis
config = GPTConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias
)
model = GPT(config, num_classes=wandb.config.num_classes)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Create datasets and dataloaders
train_dataset = SentimentDataset('data/customer_service/cleaned_train.csv',
                                 'data/customer_service/meta.pkl',
                                 max_length=block_size)
val_dataset = SentimentDataset('data/customer_service/val_split.csv',
                               'data/customer_service/meta.pkl',
                               max_length=block_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Training loop
for epoch in range(max_epochs):
    model.train()
    train_loss_total = 0.0
    train_correct = 0
    train_total = 0
    start_time = time.time()
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, loss = model(inputs, targets=labels)
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item()
        
        # Compute training accuracy for the batch
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_train_loss = train_loss_total / len(train_loader)
    train_accuracy = train_correct / train_total
    epoch_time = time.time() - start_time

    # Validation
    model.eval()
    val_loss_total = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, loss = model(inputs, targets=labels)
            val_loss_total += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_val_loss = val_loss_total / len(val_loader)
    val_accuracy = sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f} sec - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy*100:.2f}%")
    print(f"Validation Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%, F1 Score: {f1:.4f}")
    
    # Log metrics to wandb
    wandb.log({
        "epoch": epoch+1,
        "train_loss": avg_train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": avg_val_loss,
        "val_accuracy": val_accuracy,
        "f1_score": f1,
        "epoch_time_sec": epoch_time
    })

wandb.finish()
