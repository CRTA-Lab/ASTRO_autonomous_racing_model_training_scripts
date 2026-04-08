import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
from glob import glob
from os import path
import re

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext        
        self.filenames = glob(path.join(self.root_folder,"*" + self.img_ext))            
        self.totensor = transforms.ToTensor()

        # with open("stops.txt", "r") as f:
        #     self.stop_files = set(line.strip() for line in f)



        self.class_labels = ['sharp left',
                            'left',
                            'straight',
                            'right',
                            'sharp right',
                            'stop']
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        f = self.filenames[idx]      
        fname = path.basename(f)
  
        #img = cv2.imread(f)[250:, :, :]
        img = cv2.imread(f)
        
        if self.transform == None:
            img = self.totensor(img)
        else:
            img = self.transform(img)   
        
        steering = path.split(f)[-1].split(self.img_ext)[0][6:]
        steering = float(steering[1:])   
        # name = path.split(f)[-1]  # full filename (e.g. 000016_-0.070.jpg)
        # match = re.search(r'[-+]?\d*\.\d+|\d+', name)
        # steering = float(match.group())
        # print(f"Extracted steering value: {steering} from filename: {fname}")



        if steering <= -0.5:
            steering_cls = 4
        elif steering < 0:
            steering_cls = 3
        elif steering == 0:
            steering_cls = 2
        elif steering < 0.5:
            steering_cls = 1
        else:
            steering_cls = 0 

        # stop_label = 1.0 if fname in self.stop_files else 0.0
                      
        return img, steering_cls
