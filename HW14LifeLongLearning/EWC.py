import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.utils.data.sampler as sampler
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import tqdm
from tqdm import trange

class EWC(nn.Module):
    def __init__(self,model,dataloader,device,prev_grauds=[None] ) :
        super().__init__()
        self.model=model
        self.dataloader=dataloader
        self.device=device
        self.params={n:p for n,p in self.model.named_parameters() if p.requires_grad()}
        self.p_old={}
        self.previous_graud_list=prev_grauds
        self._precision_matrices=self.calculate_importance()
        for n,p in self.params.items():
            self.p_old[n]=p.clone().detach()
    
    def calculate_importance(self):
        precision_matrices={}
        for n,p in self.params.items():
            precision_matrices[n]=p.clone().detach().fill(0)
            for i in range(len(self.previous_graud_list)):
                if self.previous_graud_list[i]:
                    precision_matrices[n]+=self.previous_graud_list[i][n]
        self.model.eval()
        if self.dataloader is not None:
            number_data=len(self.dataloader)
            for data in self.dataloader:
                self.model.zero_grad()
                input=data[0].to(self.device)
                output=self.model(input)
                label=data[1].to(self.device)
                
                loss=F.nll_loss(F.softmax(output,dim=1),label)
                loss.backward()

                for n,p in self.model.named_parameters():
                    precision_matrices[n].data+=p.grad.data**2/number_data
            precision_matrices={n:p for n,p in precision_matrices.items()}
        return precision_matrices 
    
    def penalty(self,model:nn.Module):
        loss=0
        for n,p in model.parameters():
            _loss=self._precision_matrices[n]*(p-self.p_old[n])**2
            loss+=_loss.sum()
        return loss
    
    def update(self,model):
        pass


class Model(nn.Module):
  """
  Model architecture 
  1*28*28 (input) → 1024 → 512 → 256 → 10
  """
  def __init__(self):
    super(Model, self).__init__()
    self.fc1 = nn.Linear(1*28*28, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 256)
    self.fc4 = nn.Linear(256, 10)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.view(-1, 1*28*28)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.relu(x)
    x = self.fc4(x)
    return x

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Args:
  task_number = 5
  epochs_per_task = 10
  lr = 1.0e-4
  batch_size = 128
  test_size=8192

args=Args()
angle_list = [20 * x for x in range(args.task_number)]

# prepare rotated MNIST datasets.
def _rotate_image(image, angle):
  if angle is None:
    return image

  image = transforms.functional.rotate(image, angle=angle)
  return image

def get_transform(angle=None):
  transform = transforms.Compose(
                  [transforms.ToTensor(),
                   transforms.Lambda(lambda x: _rotate_image(x, angle)),
                   Pad(28)
                   ])
  return transform

class Pad(object):
  def __init__(self, size, fill=0, padding_mode='constant'):
    self.size = size
    self.fill = fill
    self.padding_mode = padding_mode
    
  def __call__(self, img):
    # If the H and W of img is not equal to desired size,
    # then pad the channel of img to desired size.
    img_size = img.size()[1]
    assert ((self.size - img_size) % 2 == 0)
    padding = (self.size - img_size) // 2
    padding = (padding, padding, padding, padding)
    return F.pad(img, padding, self.padding_mode, self.fill)

class Data():
  def __init__(self, path, train=True, angle=None):
    transform = get_transform(angle)
    self.dataset = datasets.MNIST(root=os.path.join(path, "MNIST"), transform=transform, train=train, download=True)

train_datasets = [Data('data', angle=angle_list[index]) for index in range(args.task_number)]
train_dataloaders = [DataLoader(data.dataset, batch_size=args.batch_size, shuffle=True) for data in train_datasets]

test_datasets = [Data('data', train=False, angle=angle_list[index]) for index in range(args.task_number)]
test_dataloaders = [DataLoader(data.dataset, batch_size=args.test_size, shuffle=True) for data in test_datasets]



model=Model()
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)

lll_object=EWC(model=model,dataloader=None,device=device)
lll_lambda=10000
ewc_acc=[]
prev_grads=[]
task_bar = tqdm.trange(len(train_dataloaders),desc="Task   1")

def train(model, optimizer, dataloader, epochs_per_task, lll_object, lll_lambda, test_dataloaders, evaluate, device, log_step=1):
  model.train()
  model.zero_grad()
  objective = nn.CrossEntropyLoss()
  acc_per_epoch = []
  loss = 1.0
  bar = tqdm.trange(epochs_per_task, leave=False, desc=f"Epoch 1, Loss: {loss:.7f}")
  for epoch in bar:
    for imgs, labels in tqdm.auto.tqdm(dataloader, leave=False):            
      imgs, labels = imgs.to(device), labels.to(device)
      outputs = model(imgs)
      loss = objective(outputs, labels)
      total_loss = loss
      lll_loss = lll_object.penalty(model)
      total_loss += lll_lambda * lll_loss 
      lll_object.update(model)
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()

      loss = total_loss.item()
      bar.set_description_str(desc=f"Epoch {epoch+1:2}, Loss: {loss:.7f}", refresh=True)
    acc_average  = []
    for test_dataloader in test_dataloaders: 
      acc_test = evaluate(model, test_dataloader, device)
      acc_average.append(acc_test)
    average=np.mean(np.array(acc_average))
    acc_per_epoch.append(average*100.0)
    bar.set_description_str(desc=f"Epoch {epoch+2:2}, Loss: {loss:.7f}", refresh=True)
                
  return model, optimizer, acc_per_epoch

def evaluate(model,testdataloader,device):
   model.eval()
   correct_cnt=0
   total=0
   for image,label in testdataloader:
      image,label=image.to(device),label.to(device)
      outputs=model(image)
      _,pre_label=torch.max(outputs.data,1)

      correct_cnt+=(pre_label==label.data).sum().item()
      total+=torch.ones_like(label.data).sum().item()
   return correct_cnt/total

for train_indexes in task_bar:
    # Train Each Task
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
    prev_grads.append(lll_object._precision_matrices)
    lll_object=EWC(model=model,dataloader=test_dataloaders[train_indexes],device=device,prev_grauds=prev_grads)

    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    ewc_acc.extend(acc_list)
    # Update tqdm displayer
    task_bar.set_description_str(f"Task  {train_indexes+2:2}")

# average accuracy in each task per epoch!     
print(ewc_acc)
print("==================================================================================================")

