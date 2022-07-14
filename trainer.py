#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


class Drive_Trainer(nn.Module):
  def __init__(self, model, opt="adam", lr=0.001, has_scheduler=False, device="cpu",
               log_dir = "./log"):
    super().__init__()

    self.model = model
    self.loss = nn.CrossEntropyLoss()
    self._get_optimizer(opt=opt.lower(), lr=lr)
    self.device = device

    self.has_scheduler = has_scheduler
    if self.has_scheduler:
      self._get_scheduler()
    
    self.log_dir = log_dir
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    pass

  def _get_optimizer(self, opt, lr=0.001):
    if opt == "sgd":
      self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)
    elif opt =="adam":
      self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr, weight_decay=0.01)
    elif opt == "rmsprop":
      self.optimizer = torch.optim.RMSprop(params=self.model.parameters(), lr=lr)
    else:
      raise ValueError(f"optimizer {opt} is not supported!")

  def _get_scheduler(self):
    self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=5, gamma=0.5, verbose=True)
  
  def train(self, train_loader, valid_loader, max_epoch=10, disp_epoch=1, check_point=False):
    print("==== Train Start ====")
    start_time = time.time()
    history = {"train_loss":[], "train_acc":[], "valid_loss":[], "valid_acc":[]}
    
    for e in range(max_epoch):
      print(f"Start Train Epoch {e}")
      train_loss, train_acc = self._train_epoch(train_loader)
      print(f"Start Valid Epoch {e}")
      valid_loss, valid_acc = self._valid_epoch(valid_loader)

      history["train_loss"].append(train_loss)
      history["train_acc"].append(train_acc)
      history["valid_loss"].append(valid_loss)
      history["valid_acc"].append(valid_acc)

      if self.has_scheduler:
        self.scheduler.step()
      
      if (e % disp_epoch == 0) | (e == max_epoch-1) :
        print(f"Epoch: {e}, Train loss: {train_loss:>6f}, Train acc: {train_acc:>3f}, Valid loss: {valid_loss:>6f}, Valid acc: {valid_acc:>3f}, time: {time.time()-start_time:>3f}")
        start_time = time.time()
      
        self.plot_history(history, save_name=f"{self.log_dir}/{self.model.name}_graph_epoch_{e}.png") # disp epoch마다 plotting
      
        if check_point:
          torch.save({
              'epoch' : e,
              'model_state_dict' : self.model.state_dict(),
              'optimizer_state_dict' : self.optimizer.state_dict(),
              'loss' : train_loss
          }, f"{self.log_dir}/{self.model.name}_log_epoch_{e}.pth")
        else:
          torch.save(self.model.state_dict(), f"{self.log_dir}/{self.model.name}_log_epoch_{e}.pth")  # disp epoch마다 model save
    


  def _train_epoch(self, train_loader, disp_step=10):
    epoch_loss, epoch_acc = 0, 0
    self.model.train()
    cnt = 0
    epoch_start_time = time.time()
    start_time = time.time()
    for (x, y) in train_loader:
      cnt += 1
      x = x.to(self.device)
      y = y.to(self.device)

      y_hat = self.model(x)
      loss = self.loss(y_hat, y)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      epoch_loss += loss.to("cpu").item()
      epoch_acc += (y_hat.argmax(dim=1) == y).type(torch.float).to("cpu").mean().item() # batch size 만큼의 평균
      
      
      if cnt % disp_step == 0:
        print(f"Iter: {cnt}/{len(train_loader)}, train epoch loss: {epoch_loss/(cnt):>6f}, train epoch acc: {epoch_acc/(cnt):>3f}, time: {time.time()-start_time:>3f}")
        start_time = time.time() 
      
    
    epoch_loss /= len(train_loader) # batch 개수 만큼 평균 
    epoch_acc /= len(train_loader)
    print(f"train loss: {epoch_loss:>6f}, train acc: {epoch_acc:>4f}, time: {time.time()-epoch_start_time:>3f}")

    return epoch_loss, epoch_acc 


  def _valid_epoch(self, valid_loader, disp_step=10):
    epoch_loss, epoch_acc = 0, 0
    self.model.eval()
    cnt = 0
    epoch_start_time = time.time()
    start_time = time.time()
    with torch.no_grad():
      for (x, y) in valid_loader:
        cnt += 1
        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        epoch_loss += loss.to("cpu").item()
        epoch_acc += (y_hat.argmax(dim=1) == y).type(torch.float).to("cpu").mean().item()
        
        
        if cnt % disp_step == 0:
          print(f"Iter: {cnt}/{len(valid_loader)}, valid epoch loss: {epoch_loss/(cnt):>6f}, valid epoch acc: {epoch_acc/(cnt):>3f}, time: {time.time()-start_time:>3f}")
          start_time = time.time()
        
      
    epoch_loss /= len(valid_loader)
    epoch_acc /= len(valid_loader)
    print(f"valid loss: {epoch_loss:>6f}, valid acc: {epoch_acc:>4f}, time: {time.time()-epoch_start_time:>3f}")

    return epoch_loss, epoch_acc

  
  def plot_history(self, history, save_name=None):
    fig = plt.figure(figsize=(10,5))

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history["train_loss"], color="red", label="train loss")
    ax.plot(history["valid_loss"], color="blue", label="valid loss")
    ax.set_title("Loss")
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history["train_acc"], color="red", label="train acc")
    ax.plot(history["valid_acc"], color="blue", label="valid acc")
    ax.set_title("Acc")
    ax.legend()

    plt.suptitle("Loss_Accuracy graph", fontsize=12, fontweight='bold')
    if not save_name == None:
      plt.savefig(save_name, bbox_inches='tight')
      plt.show();
    pass

  def test(self, test_loader, disp_step=10):
    print("==== Test Start ====")    
    epoch_loss, epoch_acc = 0, 0
    
    self.model.eval()
    cnt = 0
    epoch_start_time = time.time()
    start_time = time.time()
    with torch.no_grad():
      for (x, y) in test_loader:
        cnt += 1
        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.model(x)
        loss = self.loss(y_hat, y)

        epoch_loss += loss.to("cpu").item()
        epoch_acc += (y_hat.argmax(dim= 1) == y).type(torch.float).to("cpu").mean().item()

        
        if cnt % disp_step == 0:
          print(f"Iter: {cnt}/{len(test_loader)}, test epoch loss: {epoch_loss/(cnt):>6f}, test epoch acc: {epoch_acc/(cnt):>3f}, time: {time.time()-start_time:>3f}")
          start_time = time.time()
        

    epoch_loss /= len(test_loader)
    epoch_acc /= len(test_loader)

    print(f"Test loss: {epoch_loss:>6f}, Test acc: {epoch_acc:>3f}, time: {time.time()-epoch_start_time:>3f}")
    print("--------------------------------------------------------")
    print()
    print()

