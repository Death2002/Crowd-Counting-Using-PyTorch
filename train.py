# This script parses command line arguments and prepares the data
# used for training and testing the CDENet model.
import sys
import os

import warnings

# Import the CDENet model from the model.py file
from model import CDENet

# Import the save_checkpoint function from utils.py
from utils import save_checkpoint

# Import PyTorch modules
import torch
import torch.nn as nn
from torch.autograd import Variable

# Import PyTorch vision modules for loading the data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Import numpy for array manipulation
import numpy as np

# Import argparse for parsing command line arguments
import argparse

# Import json for parsing json files
import json

# Import cv2 for image processing
import cv2

# Import the dataset module from the project
import dataset

# Import time for generating random seeds
import time

# Create an ArgumentParser object to handle command line arguments
parser = argparse.ArgumentParser(description='PyTorch CDENet')

# Add an argument for the path to the train json file
parser.add_argument('train_json', metavar='TRAIN', 
                    help='path to train json')

# Add an argument for the path to the test json file
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

# Add an argument for the path to the pre-trained model
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')

# Add an argument for the GPU ID to use
parser.add_argument('gpu', metavar='GPU', type=str,
                    help='GPU id to use.')

# Add an argument for the task ID to use
parser.add_argument('task', metavar='TASK', type=str,
                    help='task id to use.')


def main():
    """
    This function parses the command line arguments, loads the json files,
    sets the random seed, creates the model, criterion and optimizer,
    and then calls the train function.
    """
    
    global args,best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    
    """
    We set some default values for certain parameters. These can be
    overridden by passing the appropriate command line arguments
    """
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs         = 400
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    
    """
    Load the json files containing the paths to the train and test images
    """
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
    """
    Set the GPU to use. The user specifies which GPU to use by
    passing the appropriate command line argument
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    """
    Set the random seed. The seed is set to the current time
    """
    torch.cuda.manual_seed(args.seed)
    
    """
    Create the model, criterion and optimizer. The model is moved to
    the GPU. 
    """
    model = CDENet()
    
    model = model.cuda()
    
    criterion = nn.MSELoss(size_average=False).cuda()
    
    # The optimizer is set to SGD with the specified learning rate,
    # momentum and weight decay. The state of the optimizer is also loaded
    # from the previous checkpoint if one is specified
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    # If a previous checkpoint is specified, then load the state of the
    # model, optimizer and best MAE score from that checkpoint. If the
    # checkpoint does not exist, then print an error message
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    # For each epoch from the starting epoch to the number of epochs,
    # adjust the learning rate according to the schedule and then call the
    # train function to train the model on the training set for one epoch.
    # After training, call the validate function to evaluate the performance
    # of the model on the validation set and then save the current state of
    # the model, optimizer and best MAE score. If the performance on the
    # validation set is the best so far, then save a separate copy of the
    # model
    for epoch in range(args.start_epoch, args.epochs):
        
        # Adjust the learning rate according to the schedule
        adjust_learning_rate(optimizer, epoch)
        
        # Train the model for one epoch
        train(train_list, model, criterion, optimizer, epoch)
        
        # Evaluate the performance of the model on the validation set
        prec1 = validate(val_list, model, criterion)
        
        # If the performance on the validation set is the best so far,
        # then save a separate copy of the model
        is_best = prec1 < best_prec1
        
        # Update the best MAE score
        best_prec1 = min(prec1, best_prec1)
        
        # Print a message showing the current best MAE score
        print(' * best MAE {mae:.3f} '.format(mae=best_prec1))
        
        # Save the current state of the model, optimizer and best MAE score
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)

def train(train_list, model, criterion, optimizer, epoch):
    """
    This function sets up a data loader to load the train images in
    batches of size args.batch_size. It also prints the number of
    samples processed, the current learning rate and the epoch number.
    
    Arguments:
        train_list (list): The list of paths to the train images
        model (nn.Module): The model we are training
        criterion (nn.Module): The loss function we are using
        optimizer (optim.Optimizer): The optimizer we are using
        epoch (int): The current epoch number
    """
    
    # Create an AverageMeter to keep track of the loss
    losses = AverageMeter()
    
    # Create an AverageMeter to keep track of the time it takes to load
    # and process each batch of data
    batch_time = AverageMeter()
    
    # Create an AverageMeter to keep track of the time it takes to load
    # each batch of data
    data_time = AverageMeter()
    
    # Create a DataLoader to load the train images in batches of size 
    # args.batch_size. The shuffle parameter is set to True so that the 
    # images are loaded in a random order.
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list, # The list of paths to the images
                       shuffle=True,  # Shuffle the images?
                       transform=transforms.Compose([ # The transformations to apply to the images
                       transforms.ToTensor(), # Convert the image to a PyTorch Tensor
                       transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalize the image using these mean and standard deviation values
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, # Are we training or validating?
                       seen=model.seen, # The number of images we have already seen
                       batch_size=args.batch_size, # The batch size
                       num_workers=args.workers), # The number of subprocesses to use for data loading
        batch_size=args.batch_size, # The batch size for the DataLoader
        pin_memory=True) # Should we use pin_memory?
    
    # Print a message showing the number of samples processed, the current
    # learning rate and the epoch number
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    # Set the model to training mode
    model.train()

    # Use the current time as the start of the batch loop
    end = time.time()

    # Loop through the batches of data
    for i, (img, target) in enumerate(train_loader):
        # Update the time it takes to load and process each batch of data
        data_time.update(time.time() - end)

        # Move the images to the GPU
        img = img.cuda()
        # Move the labels to the GPU
        target = target.cuda()

        # Reset the gradients before starting to compute the gradients for this batch
        optimizer.zero_grad()

        # Use the model to make predictions on the images
        output = model(img)

        # Compute the loss between the predicted labels and the true labels
        loss = criterion(output, target)

        # Backpropagate the gradients
        loss.backward()
        # Update the model's parameters using the gradients and the optimizer
        optimizer.step()

        # Update the average loss for this epoch
        losses.update(loss.item(), img.size(0))

        # Update the time it takes to load and process each batch of data
        batch_time.update(time.time() - end)
        # Use the current time as the start of the next batch loop
        end = time.time()

        # If we're on a batch that is a multiple of args.print_freq, print a message with the current epoch,
        # batch number, batch_time(time it takes to load and process the current batch),
        # data_time (time it takes to load the current batch), loss(average loss for the current batch)
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    
def validate(val_list, model, criterion):
    """
    This function is used to evaluate the performance of the model on the
    validation data. It uses a DataLoader to load the validation images in
    batches of size args.batch_size. The function sets the model to evaluation
    mode using the eval() function and then iterates over the validation data
    in batches. For each batch, the function computes the absolute difference
    between the predicted number of people and the true number of people. The
    function then computes the average of these differences and prints the
    resulting mean absolute error (MAE).
    
    The function returns the MAE value.
    
    Arguments:
        val_list (list): The list of paths to the validation images
        model (nn.Module): The model we are training
        criterion (nn.Module): The loss function we are using
    """
    
    print('begin test') # Print a message indicating that validation is starting
    
    # Create a DataLoader to load the validation images in batches of size 
    # args.batch_size. The shuffle parameter is set to False so that the 
    # validation images are loaded in order.
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list, # The list of paths to the images
                       shuffle=False, # Shuffle the images?
                       transform=transforms.Compose([ # The transformations to apply to the images
                           transforms.ToTensor(), # Convert the image to a PyTorch Tensor
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalize the image using these mean and standard deviation values
                                     std=[0.229, 0.224, 0.225])
                       ]), 
                       train=False), # Are we training or validating?
        batch_size=args.batch_size, # The batch size for the DataLoader
        num_workers=args.workers, # The number of subprocesses to use for data loading
        pin_memory=True) # Should we use pin_memory?
    
    # Set the model to evaluation mode using the eval() function. This is a good
    # practice when testing since it makes the code more readable and helps
    # prevent bugs.
    model.eval()
    
    mae = 0 # Initialize the MAE to zero
    for img, target in test_loader: # Iterate over the validation data in batches
        img = img.cuda() # Move the images to the GPU
        target = target.cuda() # Move the labels to the GPU
        
        # Use the model to make predictions on the images
        output = model(img).detach()
        
        # Compute the absolute difference between the predicted labels and the
        # true labels
        mae += abs(output.sum() - target.sum()).item()
        
    mae = mae / len(test_loader) # Compute the average of the absolute differences
    
    print(' * MAE {mae:.3f} '.format(mae=mae)) # Print the MAE
    
    return mae # Return the MAE value
        
def adjust_learning_rate(optimizer, epoch):
    """
    Adjust the learning rate at the given epoch using the steps and scales 
    arguments provided from the command line. The learning rate is initially set
    to the initial learning rate (args.lr) and then decayed by a factor of 10 
    every 30 epochs. The number of times the learning rate is decayed is given
    by the length of args.steps. The function uses NumPy's searchsorted function
    to determine which step the current epoch corresponds to and then uses that
    index to get the appropriate scaling factor from args.scales. The function 
    then sets the learning rate for all parameter groups in the optimizer to the 
    product of the initial learning rate and the appropriate scaling factor.

    Arguments:
        optimizer (Optimizer): The optimizer to adjust the learning rate for
        epoch (int): The current epoch
    """
    
    # Determine which step the current epoch corresponds to using searchsorted
    decay_step = np.searchsorted(args.steps, epoch, side='right')
    
    # If this is not the first step, then decay the learning rate
    if decay_step > 0:
        # Get the appropriate scaling factor from args.scales
        scale = args.scales[decay_step - 1]
        
        # Set the learning rate for all parameter groups to the product of the
        # initial learning rate and the scaling factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * scale
    
    # Otherwise, do not decay the learning rate
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        
class AverageMeter:
    """
    Class for computing and storing the average and current value
    
    The class has an instance variable for each of the following:
        - current value (val)
        - average value over a series of data points (avg)
        - sum of all data points observed (sum)
        - total number of data points observed (count)
    
    The class has one method:
        - update: updates the value of the instance variables based on the
                  current value and the number of data points to add
    """
    def __init__(self):
        """
        Constructor for the AverageMeter class
        
        Initializes all instance variables to 0
        """
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        """
        Updates the values of the instance variables based on the current value
        and the number of data points to add
        
        Arguments:
            val: the current value
            n: the number of data points to add (default is 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    main()