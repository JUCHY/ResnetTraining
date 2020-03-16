import json
import random
from glob import glob
from PIL import Image
import torchvision
from torch import device,cuda
from torch import argmax
import numpy as np
import torch.nn as nn
from torchvision import transforms, utils
import torch.utils.data as data
from torchvision.models import alexnet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


alexnet =  alexnet(pretrained=True)

class ImageNet(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.folder_paths = glob("{}/*/".format(self.path))
        self.json_path = "{}/imagenet_class_index.json".format(self.path)

        with open("{}/imagenet_class_index.json".format(self.path), "r") as f:
            self.lbl_dic = json.load(f)
        self.lbl_dic = {v[0]: int(k) for k, v in self.lbl_dic.items()}
        #print(self.lbl_dic)

        self.img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.imgs = []
        self.lbls = []
        print(self.folder_paths)
        for folder_path in self.folder_paths:
            print(folder_path.split("\\")[-2])
            image_paths = glob("{}/*".format(folder_path))
            self.imgs += image_paths
            self.lbls += [self.lbl_dic[folder_path.split("\\")[-2]]] * len(image_paths)
        #print(self.lbls)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        img = self.img_transforms(img)
        lbl = self.lbls[index]
        return img, lbl

    def __len__(self):
        return len(self.imgs)
    


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    x = random.randint(0,1000000)
    plt.savefig('sgdacc'+str(x)+'.png')
    plt.show()


# get some random training images

if __name__ == "__main__":
    # Your code goes here
    
    
    dataset = ImageNet(path="./imagenet_12")
    dataloader = DataLoader(dataset, batch_size=1,shuffle=True, num_workers=4)
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(labels)
    #print([dataset.lbl_dic[int(x)] for x in labels])
    device = device("cuda:0" if cuda.is_available() else "cpu")
    print(device)
    alexnet.eval()
    alexnet.to(device)
    correct = 0
    total = 0
    for i, (imgs, lbls) in enumerate(dataloader):
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        output = alexnet(imgs)
        predicted = argmax(output.data)
        total += lbls.size(0)
        correct += (predicted == lbls).sum().item()
        
        if(i==len(dataloader)-1):
            print("This works motherfucker")
        
        
        print(predicted)
        print(lbls)
        print(correct)
        print(total)
        
      
    print("This much, {}% , of the dataset was correct".format((correct/total)*100))
    
    pass
