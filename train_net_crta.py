import torch # type: ignore
import torchvision.transforms as transforms # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader # pyright: ignore[reportMissingImports]
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np
import sklearn.metrics as metrics

import matplotlib.pyplot as plt

from steerDS_crta import SteerDataSet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#######################################################################################################################################
####     This tutorial is adapted from the PyTorch "Train a Classifier" tutorial                                                   ####
####     Please review here if you get stuck: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html                   ####
#######################################################################################################################################
torch.manual_seed(0)

#Helper function for visualising images in our dataset
def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    rgbimg = npimg[:,:,::-1]
    plt.imshow(rgbimg)
    plt.show()

#######################################################################################################################################
####     SETTING UP THE DATASET                                                                                                    ####
#######################################################################################################################################

#transformations for raw images before going to CNN
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((40, 60)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

script_path = os.path.dirname(os.path.realpath(__file__))

###################
## Train dataset ##
###################


train_ds = SteerDataSet(os.path.join(script_path, '../..', 'images_fast', 'train'), '.jpg', transform)
print("The train dataset contains %d images " % len(train_ds))

#data loader nicely batches images for the training process and shuffles (if desired)
trainloader = DataLoader(train_ds,batch_size=8,shuffle=True)
all_y = []
for S in trainloader:
    im, y = S    
    all_y += y.tolist()

# all_stop = []
# for S in trainloader:
#     _, _, stop = S
#     all_stop += stop.tolist()

# all_stop = np.array(all_stop)

# stop_vals, stop_counts = np.unique(all_stop, return_counts=True)

# plt.bar(stop_vals, stop_counts, width=0.4)
# plt.xticks([0, 1], ['No Stop', 'Stop'])
# plt.xlabel('Stop Label')
# plt.ylabel('Counts')
# plt.title('Training Dataset – Stop Labels')
# plt.show()


print(f'Input to network shape: {im.shape}')

#visualise the distribution of GT labels
all_lbls, all_counts = np.unique(all_y, return_counts = True)


plt.bar(all_lbls, all_counts, width = (all_lbls[1]-all_lbls[0])/2)
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.xticks(all_lbls)
plt.title('Training Dataset')
plt.show()

# # visualise some images and print labels -- check these seem reasonable
# example_ims, example_lbls, stop_lbls = next(iter(trainloader))
# print(' '.join(f'{example_lbls[j]}' for j in range(len(example_lbls))))
# imshow(torchvision.utils.make_grid(example_ims))


########################
## Validation dataset ##
########################

val_ds = SteerDataSet(os.path.join(script_path, '../..', 'images_fast', 'val'), '.jpg', transform)
print("The val dataset contains %d images " % len(val_ds))

#data loader nicely batches images for the training process and shuffles (if desired)
valloader = DataLoader(val_ds,batch_size=1)
all_y = []
for S in valloader:
    im, y = S   
    all_y += y.tolist()

print(f'Input to network shape: {im.shape}')

#visualise the distribution of GT labels
all_lbls, all_counts = np.unique(all_y, return_counts = True)
plt.bar(all_lbls, all_counts, width = (all_lbls[1]-all_lbls[0])/2)
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.xticks(all_lbls)
plt.title('Validation Dataset')
plt.show()



#######################################################################################################################################
####     CONFIGURE CLASSIFICATION MODEL ARCHITECTURE                                                                               ####
#######################################################################################################################################

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1   = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc_shared = nn.Linear(1344, 256)

        self.fc_drive1 = nn.Linear(256, 128)  
        self.fc_drive2 = nn.Linear(128,5) 
        # self.fc_stop1  = nn.Linear(256, 64)  
        # self.fc_stop2  = nn.Linear(64, 1) 

    def forward(self, x):

        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)

        x = self.relu(self.fc_shared(x))

        drive_logits = self.fc_drive1(x)
        drive_logits = self.relu(drive_logits)
        drive_logits = self.fc_drive2(drive_logits)
        
        # stop_logit   = self.fc_stop1(x)
        # stop_logit = self.relu(stop_logit)
        # stop_logit   = self.fc_stop2(stop_logit)
        return drive_logits
    

net = Net()

# num_stop = stop_counts[stop_vals == 1][0]
# num_nostop = stop_counts[stop_vals == 0][0]

# pos_weight = torch.tensor([num_nostop / num_stop])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)


#######################################################################################################################################
####     TRAINING HYPERPARAMETERS		                                                                                   ####
#######################################################################################################################################

drive_criterion = nn.CrossEntropyLoss()
# stop_criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

optimizer = optim.Adam(net.parameters(), lr=1e-3)

NUM_EPOCHS = 6
# LAMBDA_STOP = 0.5


#######################################################################################################################################
####     TRAINING 				                                                                                   ####
#######################################################################################################################################

for epoch in range(NUM_EPOCHS):

    
    net.train()
    running_loss = 0.0

    for imgs, drive_lbls in trainloader:
        imgs = imgs.to(device)
        drive_lbls = drive_lbls.to(device)
        # stop_lbls = stop_lbls.to(device)

        optimizer.zero_grad()
        print(imgs.shape)
        drive_logits = net(imgs)

        
        # loss_stop = stop_criterion(
        #     stop_logits.squeeze(1),
        #     stop_lbls
        # )

        
        # mask = (stop_lbls == 0)

        loss_drive = drive_criterion(
            drive_logits,
            drive_lbls
        )

        loss = loss_drive 
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"[Epoch {epoch+1}] Train loss: {running_loss/len(trainloader):.4f}")

    # VALIDATION
    net.eval()
    val_drive_correct = 0
    val_drive_total = 0
    # val_stop_correct = 0
    # val_stop_total = 0

    with torch.no_grad():
        for imgs, drive_lbls in valloader:
            imgs = imgs.to(device)
            drive_lbls = drive_lbls.to(device)
            # stop_lbls = stop_lbls.to(device)
            

            drive_logits = net(imgs)

            
            # stop_pred = (torch.sigmoid(stop_logits) > 0.9).float()
            # val_stop_correct += (stop_pred.squeeze() == stop_lbls).sum().item()
            # val_stop_total += stop_lbls.size(0)

            
            # mask = (stop_lbls == 0)
            # if mask.any():
            preds = torch.argmax(drive_logits, dim=1)
            val_drive_correct += (preds == drive_lbls).sum().item()
            val_drive_total += drive_lbls.size(0)

    print(
        f"  Val drive acc: {val_drive_correct/max(val_drive_total,1):.3f} | "
        # f"Val stop acc: {val_stop_correct/val_stop_total:.3f}"
    )
print('Finished Training')

#Save the trained model
torch.save(net.state_dict(), "steer_net_2.pth")


#######################################################################################################################################
####     VALIDATION 				                                                                                   ####
#######################################################################################################################################

net.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for imgs, drive_lbls in valloader:
        imgs = imgs.to(device)
        drive_lbls = drive_lbls.to(device)
        # stop_lbls = stop_lbls.to(device)

        drive_logits = net(imgs)

        # Only evaluate steering when NOT stop
        # mask = (stop_lbls == 0)

        # if mask.any():
        preds = torch.argmax(drive_logits, dim=1)

        y_true.extend(drive_lbls.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "sharp left",
        "left",
        "straight",
        "right",
        "sharp right"
    ]
)

disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Steering Confusion Matrix (Validation)")
plt.show()
