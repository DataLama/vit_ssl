import os
import math
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from omegaconf import OmegaConf
import torch
from torch import nn
from pytorch_lightning import seed_everything

from transformers import get_scheduler
from pytorch_lightning import seed_everything

from src.pt.data import ImageClassificationDataModule
from src.pt.model import ViT

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def simple_train(cfg):
    ## load dataloader
    dm = ImageClassificationDataModule(**OmegaConf.to_container(cfg.data))
    dm.setup()
    
    tr_dl = dm.train_dataloader()
    vl_dl = dm.val_dataloader()
    
    ## load model and loss
    model = ViT(**OmegaConf.to_container(cfg.model), num_classes=dm.num_classes)
    
    ## load criterion
    criterion = nn.CrossEntropyLoss()
    
    ## get optimizer and scheduler
    num_update_steps_per_epoch = math.ceil(len(tr_dl) / cfg.train.gradient_accumulation_steps)
    cfg.train.max_train_steps = int(cfg.train.num_train_epochs * num_update_steps_per_epoch)
    
    # optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.train.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.train.learning_rate)
    
    # scheduler
    lr_scheduler = get_scheduler(
        name=cfg.train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.train.num_warmup_steps,
        num_training_steps=cfg.train.max_train_steps,
    )
    
    
    ## 
    print("Run Trainining")
    print(f"  Total train batch size  = {cfg.data.batch_size}")
    model.to(DEVICE)
    losses = 0.0
    for epoch in range(int(cfg.train.num_train_epochs)):
        model.train()
        for step, batch in enumerate(tr_dl):
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            logits = model(pixel_values)
            loss = criterion(logits.view(-1, dm.num_classes), labels.view(-1))
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # print statistics
            losses += loss.detach().item()
            if step % 1000 == 999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {step + 1:5d}] loss: {losses / 2000:.3f}')
                losses = 0.0
    return model

if __name__=="__main__":
    cfg = OmegaConf.load('conf/simple_vit.yaml')
    seed_everything(cfg.data.seed)
    model = simple_train(cfg)
    torch.save(model.state_dict(), 'model.pth')