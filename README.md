# ASTRO autonomous racing training scripts

This is the ASTRO autonomous racing training folder with training scripts, used in **astro_autonomous_racing** ROS2 package.

Includes:
- **```splitter.py```** :


```splitter.py``` splits the images dataset into ```images/train``` and ```images/val```.

Usage:

```bash
cd ~/astro_ws/
python3 ASTRO_autonomous_racing_model_training_scripts/splitter.py
```

- **```steerDS_crta.py```** is helper script for ```train_net_crta.py```:

```steerDS_crta.py``` script describes the remappingg between the steering ```{cmd_vel.angular.z}```  to ``` self.class_labels = ['sharp left', 'left', 'straight', 'right', 'sharp right', 'stop']```

- **```train_net_crta.py```** :

Model training script - ```train_net_crta.py``` have multiple segments:
- **```SETTING UP THE DATASET```** - This part of the script sets up the transformations on raw images from /train /val folders to be prepared as inputs to the model. It sets up the train and validation datasets and displays the dataset's class balance and the start. **Do not change this segment**
- **```CONFIGURE CLASSIFICATION MODEL ARCHITECTURE```** - This part confiures the model architecture. Here **you** should specify the architecture. For the reference use this [PyTorch Classifer Tutorial]((https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html))
- **```TRAINING HYPERPARAMETERS```** - Here **you** specify the model training hyperparameters: Loss function as criterion, optimizer, number of epochs.
- **```TRAINING```** - Model train segment. Here **you** specify the name of saved trained model. ```<model_name>.pth```
- **```VALIDATION```** - Validation segment, displays the confusion matrix. **Do not chnage this segment**

Start model training procedure:
```bash
cd ~/astro_ws/
python3 ASTRO_autonomous_racing_model_training_scripts/train_net_crta.py
```