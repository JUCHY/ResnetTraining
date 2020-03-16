# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:45:15 2020

@author: joshu
"""

import json
import time
import random
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.models import resnet18
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import argmax
from torch.utils.data import DataLoader



class ImageNet(data.Dataset):
    def __init__(self, path,path2):
        self.path = path
        self.path2 = path2
        self.folder_paths = glob("{}/{}/*/".format(self.path,self.path2))
        self.json_path = "{}/imagenet_class_index.json".format(self.path)

        with open("{}/imagenet_class_index.json".format(self.path), "r") as f:
            self.lbl_dic = json.load(f)
        self.lbl_dic = {v[0]: int(k) for k, v in self.lbl_dic.items()}

        self.img_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.imgs = []
        self.lbls = []
        print(self.folder_paths)
        for folder_path in self.folder_paths:
            image_paths = glob("{}/*".format(folder_path))
            self.imgs += image_paths
            self.lbls += [self.lbl_dic[folder_path.split("\\")[-2]]] * len(image_paths)
        
        pass
    def __getitem__(self,index):
        img = Image.open(self.imgs[index]).convert("RGB")
        img = self.img_transforms(img)
        lbl = self.lbls[index]
        if lbl==283:
            lbl=0
        if lbl==248:
            lbl=1
        return img,lbl
    
    def __len__(self):
        return len(self.imgs)
    
    
    
"""
Notes: lbls, taken from image name to determine what category it falls into,
image is stored in x,
and lbl of image is stored in x , to store for verification
Ask about labels in class today
how to use gpu

"""
    
    
class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        #create a headless ResNet
        self.model = resnet18(pretrained=False)
        self.resnet = nn.Sequential(*list(self.model.children())[:-1])
        self.linear = nn.Linear(512,2)
        #self.linear = nn.linear(512,2)
        
        
        

    def forward(self, x):
        
        #pass input to headless ResNet 
        # batch_size = # of images, channels(rgb), x, y
        x = self.resnet.forward(x)
        x = x.view(-1,512)
        return self.linear(x)
    
        
    
    
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
    
def RMS( train_dataloader, val_dataloader, lrx=0.01, momentum=0, a=0.99, decay=0):
        
    model = net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr = lrx, alpha=a, weight_decay=decay)
    accuracy = []
    lossgraph = []
    epochs = []
    for epoch in range(50):
        epochs.append(epoch)
        #trainining
        t = time.time()
        
        #set modedl to train model
        model.train()
        running_loss = 0.0
        for batch_idex, (imgs, lbls) in enumerate(train_dataloader):
            # If you are using gpu, move data to gpu
            imgs = imgs.to(device)
            lbls = lbls.to(device)
           #zero paramaeter gradients
            optimizer.zero_grad()
            #get loss and do backprop
            output = model(imgs)
            loss = loss_fn(output, lbls)
            loss.backward()
            optimizer.step()
            
            running_loss+=loss.item()
            if batch_idex==len(train_dataloader)-1:
                print("Epoch {} train: {}/{} loss: {:.5f} ({:3f}s)".format(
                        epoch+1, batch_idex+1, len(train_dataloader), running_loss/len(train_dataloader), time.time()-t), end="\r")
                running_loss = 0.0
        
    
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for i, (imgs, lbls) in enumerate(val_dataloader):
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                output = model(imgs)
                loss = loss_fn(output, lbls)
                val_loss+=loss.item()
                predicted = argmax(output.data)
                total += lbls.size(0)
                correct += (predicted == lbls).sum().item()                    
          
        print("This much, {}% , of the dataset was correct".format((correct/total)*100))
        accuracy.append((correct/total))
        lossgraph.append(val_loss/(len(val_dataloader)))
        #plot loss graph
    plt.plot(epochs, lossgraph, label = str(lrx))
    plt.xlabel('Epochs')
    plt.ylabel('Running Loss')
    plt.title('Loss Graph')
    plt.xticks(epochs)
    plt.legend()
    x = random.randint(0,100000)
    plt.savefig('rmsloss'+str(x)+'.png')
    plt.show()
    #plot accuracy graph
    plt.plot(epochs, accuracy, label = str(lrx))
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy Graph')
    plt.legend()
    y = random.randint(0,100000)
    plt.savefig('rmsacc'+str(y)+'.png')
    plt.show()
    
if __name__ == "__main__":

    torch.manual_seed(200)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    
    #use cuda if available

    #create a dataset
    train_dataset = ImageNet(path="./imagenet_12",path2="imagenet_12_train")
    val_dataset = ImageNet(path="./imagenet_12",path2="imagenet_12_val")

   
    #create a test dataset
    train_dataloader = DataLoader(train_dataset, batch_size=4,shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=1,shuffle=True, num_workers=4)

    #dataiter = iter(val_dataloader)
    #images, labels = dataiter.next()
   # # show images
  #  imshow(utils.make_grid(images))
 #   print(labels)
        
    #define model, loss function, optimizer
    
    model = net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # look up later: crossentropyloss

    lrx = [0.1, 0.01, 0.001]
    accuracy = []
    lossgraph = []
    idx = 0
    for x in lrx:
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr = x)
        epochs = []
        accuracy.append([])
        lossgraph.append([])
        for epoch in range(50):

            #trainining
            epochs.append(epoch)
            t = time.time()
            
            #set modedl to train model
            model.train()
            running_loss = 0.0
            for batch_idex, (imgs, lbls) in enumerate(train_dataloader):
                # If you are using gpu, move data to gpu
                imgs = imgs.to(device)
                lbls = lbls.to(device)
               #zero paramaeter gradients
                optimizer.zero_grad()
                #get loss and do backprop
                output = model(imgs)
                loss = loss_fn(output, lbls)
                loss.backward()
                optimizer.step()
                
                running_loss+=loss.item()
                if batch_idex==len(train_dataloader)-1:
                    print("Epoch {} train: {}/{} loss: {:.5f} ({:3f}s)".format(
                            epoch+1, batch_idex+1, len(train_dataloader), running_loss/len(train_dataloader), time.time()-t), end="\r")
                    running_loss = 0.0
            
        
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            with torch.no_grad():
                for i, (imgs, lbls) in enumerate(val_dataloader):
                    imgs = imgs.to(device)
                    lbls = lbls.to(device)
                    output = model(imgs)
                    loss = loss_fn(output, lbls)
                    val_loss+=loss.item()
                    predicted = argmax(output.data)
                    total += lbls.size(0)
                   # print(total)
                    correct += (predicted == lbls).sum().item()                    
              
                print("This much, {}% , of the dataset was correct".format((correct/total)*100))
                accuracy[idx].append((correct/total))
                lossgraph[idx].append(val_loss/(len(val_dataloader)))
        idx+=1
        #plot loss graph
    #print(lossgraph)
    #rint(epochs)
    #print(accuracy)
    for i, val in enumerate(lrx):
        plt.plot(lossgraph[i], label=str(val))
    plt.xlabel('Epochs')
    plt.xticks(epochs)
    plt.ylabel('Running Loss')
    plt.title('Loss Graph')
    plt.legend()
    plt.savefig('sgdloss.png')
    plt.show()
    #plot accuracy graph
    for i, val in enumerate(lrx):
        plt.plot(accuracy[i], label=str(val))
    plt.xlabel('Epochs')
    plt.xticks(epochs)
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy Graph')
    plt.legend()
    plt.savefig('sgdacc.png')
    plt.show()

            
            
            
            
        



    
#    #test data with dataset  
#    test_dataset = ImageNet(path="./imagenet_12",path2="imagenet_12_val")
#    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
#    
#    model.eval()
#    correct = 0
#    total = 0
#    for i, (imgs, lbls) in enumerate(test_dataloader):
#        output = model(imgs)
#        predicted = argmax(output.data)
#        total += lbls.size(0)
#        correct += (predicted == lbls).sum().item()        
#      
#    print("This much, {}% , of the dataset was correct".format((correct/total)*100))
#    
#    
        
    print("With HPM Set 1")
    #default values
    RMS(train_dataloader,val_dataloader)
    print("With HPM Set 2")
    RMS(train_dataloader, val_dataloader, 0.01, 0.9, 0.99, 0.01)
    print("With  HPM Set 3")
    RMS(train_dataloader,val_dataloader, 0.1, 0.9, 0.5, 0.1)
    
    
    pass




#what does optimizer do?
#asak about net module