import torch
import torch.nn as nn


def training(epoch, model, optimizer, scheduler, criterion, device, train_loader):
  model.train()
  avg_loss = 0.0
  av_loss=0.0
  total=0
  for batch_num, (feats, labels) in enumerate(train_loader):
      feats, labels = feats.to(device), labels.to(device)
      
      optimizer.zero_grad()

      outputs = model(feats)


      loss = criterion(outputs, labels.long())
      loss.backward()
      
      optimizer.step()
      
      avg_loss += loss.item()
      av_loss += loss.item() 
      total += len(feats) 
      # if batch_num % 10 == 9:
      #     print('Epoch: {}\tBatch: {}\tAv-Loss: {:.4f}'.format(epoch+1, batch_num+1, av_loss/10))
      #     av_loss = 0.0

      torch.cuda.empty_cache()
      del feats
      del labels
      del loss

  del train_loader


def validate(epoch, model, criterion, device, data_loader):
  with torch.no_grad():
      model.eval()
      running_loss, accuracy,total  = 0.0, 0.0, 0

      
      for i, (X, Y) in enumerate(data_loader):
          
          X, Y = X.to(device), Y.to(device)
          output= model(X)
          loss = criterion(output, Y.long())

          _,pred_labels = torch.max(F.softmax(output, dim=1), 1)
          pred_labels = pred_labels.view(-1)
          
          accuracy += torch.sum(torch.eq(pred_labels, Y)).item()

          running_loss += loss.item()
          total += len(X)

          torch.cuda.empty_cache()
          
          del X
          del Y
      
      return running_loss/total, accuracy/total
