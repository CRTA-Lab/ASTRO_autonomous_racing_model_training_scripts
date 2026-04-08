import os 
import numpy as np
import shutil


#Create the train and validation folders
os.makedirs("images/train", exist_ok=True)
os.makedirs("images/val", exist_ok=True)

#List all the images from images folder and remapp their cmd_vel.angular.z to the classifications [0,1,2,3,4] that represents [sharp left, left, straight, right, sharp right]
data = []
for img_id in os.listdir("images/images"):
    data.append([img_id, img_id.split(".")[1]])
    # data.append([img_id.split(".")[0], img_id.split(".")[1]])

labels = []
for data_id in data:
    if "-" in data_id[0].split(".")[0]:
        if float(data_id[1]) >= 50:
            labels.append(0)
        if float(data_id[1]) > 0 and float(data_id[1]) < 50:
            labels.append(1)
    else:
        if float(data_id[1]) >= 50:
            labels.append(4)
        if float(data_id[1]) > 0 and float(data_id[1]) < 50:
            labels.append(3)
    if float(data_id[1]) == 0:
        labels.append(2)

names = [i[0] for i in data]

#Balance the dataset
min_indices = round(min(labels.count(0), labels.count(1), labels.count(2), labels.count(3), labels.count(4)) * 0.8)

labels = np.array(labels)
cls_0_ind = np.where(labels == 0)[0][:min_indices]
cls_1_ind = np.where(labels == 1)[0][:min_indices]
cls_2_ind = np.where(labels == 2)[0][:min_indices]
cls_3_ind = np.where(labels == 3)[0][:min_indices]
cls_4_ind = np.where(labels == 4)[0][:min_indices]

names_balanced = []
for i in cls_0_ind:
    names_balanced.append(names[i])
for i in cls_1_ind:
    names_balanced.append(names[i])
for i in cls_2_ind:
    names_balanced.append(names[i])
for i in cls_3_ind:
    names_balanced.append(names[i])
for i in cls_4_ind:
    names_balanced.append(names[i])


#Copy the images to the train/val folders
for i in names_balanced:
    shutil.copy(os.path.join("images/images", i), os.path.join("images/val", i))    

for i in os.listdir("images/images"):
    if i not in names_balanced:
        shutil.copy(os.path.join("images/images", i), os.path.join("images/train", i))
