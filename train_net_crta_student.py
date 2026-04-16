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
print(script_path)

###################
## Train dataset ##
###################


train_ds = SteerDataSet(os.path.join(script_path, '..', 'images', 'train'), '.jpg', transform)
print("The train dataset contains %d images " % len(train_ds))

# --------------------------------------------------------------------------------------------------------------------
# TODO 1 (students): choose training hyperparameters.
# Suggested experiments:
#   - batch size
#   - optimizer type
#   - learning rate
#   - number of epochs
# --------------------------------------------------------------------------------------------------------------------
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 6

#data loader nicely batches images for the training process and shuffles (if desired)
trainloader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
all_y = []
for S in trainloader:
    im, y = S    
    all_y += y.tolist()




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

val_ds = SteerDataSet(os.path.join(script_path, '..', 'images', 'val'), '.jpg', transform)
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

        # ----------------------------------------------------------------------------------------------------------------
        # TODO 2 (students): complete / improve the CNN architecture.
        # Current template: two convolutional blocks and a classifier head.
        # Ideas to try:
        #   - change the number of filters
        #   - add a third conv layer
        #   - add dropout
        # ----------------------------------------------------------------------------------------------------------------

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1   = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc_shared = nn.Linear(1344, 256)

        self.fc_drive1 = nn.Linear(256, 128)  
        self.fc_drive2 = nn.Linear(128,5) 

    def forward(self, x):

        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)

        x = self.relu(self.fc_shared(x))

        drive_logits = self.fc_drive1(x)
        drive_logits = self.relu(drive_logits)
        drive_logits = self.fc_drive2(drive_logits)
        
        return drive_logits
    

net = Net()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)


#######################################################################################################################################
####     TRAINING HYPERPARAMETERS		                                                                                   ####
#######################################################################################################################################
# --------------------------------------------------------------------------------------------------------------------
# TODO 3 (students): choose a suitable loss function for 5-class classification.
# Question to answer in your report:
#   Why is this loss appropriate for steering labels: hard left / left / straight / right / hard right?
# --------------------------------------------------------------------------------------------------------------------
drive_criterion = nn.CrossEntropyLoss()

# Optional extension for students:
# If your dataset is imbalanced, compute class weights from the training labels and pass them to CrossEntropyLoss.

# --------------------------------------------------------------------------------------------------------------------
# TODO 4 (students): choose and justify the optimizer.
# Examples to compare: Adam, SGD with momentum
# --------------------------------------------------------------------------------------------------------------------


optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

NUM_EPOCHS = NUM_EPOCHS


#######################################################################################################################################
####     TRAINING 				                                                                                   ####
#######################################################################################################################################

for epoch in range(NUM_EPOCHS):

    
    net.train()
    running_loss = 0.0

    for imgs, drive_lbls in trainloader:
        imgs = imgs.to(device)
        drive_lbls = drive_lbls.to(device)

        optimizer.zero_grad()

        drive_logits = net(imgs)

        

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

    with torch.no_grad():
        for imgs, drive_lbls in valloader:
            imgs = imgs.to(device)
            drive_lbls = drive_lbls.to(device)
            

            drive_logits = net(imgs)

            
            preds = torch.argmax(drive_logits, dim=1)
            val_drive_correct += (preds == drive_lbls).sum().item()
            val_drive_total += drive_lbls.size(0)

    print(
        f"  Val drive acc: {val_drive_correct/max(val_drive_total,1):.3f} | "
    )
    # ----------------------------------------------------------------------------------------------------------------
    # TODO 5 (students): add another validation metric.
    # Suggestions:
    #   - per-class accuracy
    #   - macro F1-score
    #   - precision / recall for each class
    # ----------------------------------------------------------------------------------------------------------------
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

        drive_logits = net(imgs)


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

# --------------------------------------------------------------------------------------------------------------------
# TODO 6 (students): write a short discussion based on the confusion matrix.
# Questions to think about:
#   - Which steering classes are easiest and hardest?
#   - Does the model confuse left with hard left or right with hard right?
#   - Is the class "straight" over-represented?
#   - What would you change next: data, preprocessing, model, or training setup?
# --------------------------------------------------------------------------------------------------------------------