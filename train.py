import argparse
import json
import os
import sys
from collections import OrderedDict

import torch
from torch import nn, optim
from torchvision import datasets, models, transforms

def get_input_args():
    parser = argparse.ArgumentParser(description='Train a deep learning model on a flower dataset')
    parser.add_argument('data_dir', type=str, help='Path to to the dataset directory')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the checkpoint (default is current directory)')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg13', 'vgg16','resnet18'], help='Model architecture to use (default is resnet18)')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate for the optimizer (default is 0.003)')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier (default is 512)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model (default is 20)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')

    return parser.parse_args()

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    #same as in part 1
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),  
                                        transforms.RandomHorizontalFlip(),  
                                        transforms.ToTensor(),  
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

    valid_test_transforms = transforms.Compose([transforms.Resize(256), 
                                                transforms.CenterCrop(224),  
                                                transforms.ToTensor(),  
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])

 
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)

    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

    dataloaders = {'train': train_loader, 'valid': valid_loader }

    dataset_sizes = {'train': len(train_dataset), 'valid': len(valid_dataset)}

    class_to_idx = train_dataset.class_to_idx

    return dataloaders, dataset_sizes, class_to_idx



def build_model(arch='resnet18', layer_1_hidden_units=256, layer_2_hidden_units=128):
    #create model based on arch, also nb of input features depends on it
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_features = 25088  
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_features = 25088
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_features = 512 
    else:
        raise ValueError(f"Architecture '{arch}' is not supported. Choose 'vgg13', 'vgg16', or 'resnet18'.")

   # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier 
    #i changed a bit from part 1 (without dropout) and made 2 hidden layers
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, layer_1_hidden_units)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(layer_1_hidden_units, layer_2_hidden_units)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(layer_2_hidden_units, 102)),  
        ('output', nn.LogSoftmax(dim=1))
    ]))

    
    if arch in ['vgg13', 'vgg16']:
        model.classifier = classifier
    elif arch == 'resnet18':
        model.fc = classifier  #like part 1 since ResNet uses 'fc' instead of 'classifier'

    return model


def validation(model, validloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    accuracy = 0
    loss = 0

    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss += criterion(output, labels).item()
            preds = output.argmax(dim=1)
            accuracy += (preds == labels).type(torch.FloatTensor).mean().item()

    average_loss = loss / len(validloader)
    average_accuracy = accuracy / len(validloader)
    return average_loss, average_accuracy

def train_model(model, trainloader, validloader, criterion, optimizer, epochs, device):
    model.to(device)
    #this is very similar to part 1 so i will copy and paste it and edit slightly!
    running_loss = 0 # Accumulate the loss to print later
     
    for epoch in range(epochs):
        model.train()  # Put the model in training mode for dropout to work
        for inputs, labels in trainloader:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # VIP: Clear any previously accumulated gradients
            
            # Forward pass
            logps = model(inputs)
            
            # Calculate loss between pred(logps) and true labels
            loss = criterion(logps, labels)
            
            # Backward pass/backpropagation
            loss.backward()
            
            # Update the weights based on the calculated gradients
            optimizer.step()
            
            running_loss += loss.item()


            #this part is new
            # Perform validation: call the function above!
            valid_loss, valid_accuracy = validation(model, validloader, criterion, device)

            print(f"Epoch {epoch+1}/{epochs}.. "
                f"Training Loss: {running_loss/len(trainloader):.3f}.. "
                f"Validation Loss: {valid_loss:.3f}.. "
                f"Validation Accuracy: {valid_accuracy:.3f}")

        print("Training complete.")
        return model


def save_checkpoint(model, save_dir, class_to_idx, arch, layer_1_hidden_units, layer_2_hidden_units, learning_rate, epochs, optimizer):

    checkpoint = {
        #now we have to save the arch too
        'arch': arch,
        'layer_1_hidden_units': layer_1_hidden_units,
        'layer_2_hidden_units': layer_2_hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'optimizer_state_dict': optimizer.state_dict()
    }

    # first make sure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'checkpoint.pth')

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}!!")

def main():
    #Now we do as requested!
    # Parse command line arguments
    args = get_input_args()

    # Load the data
    trainloader, validloader, class_to_idx = load_data(args.data_dir)

    # Build the model and use the args
    model = build_model(args.arch, args.layer_1_hidden_units, args.layer_2_hidden_units)

    # Define loss function
    criterion = nn.NLLLoss()

    # Define optimizer based on architecture
    if args.arch == 'resnet18':
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    else:  #meaning 'vgg13' or 'vgg16'
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Set device to GPU if requested and available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    if args.gpu and not torch.cuda.is_available():
        print("GPU requested but not available. Using CPU instead.")

    # Train the model
    trained_model = train_model(model, trainloader, validloader, criterion, optimizer, args.epochs, device)

    # finally, Save the checkpoint!
    save_checkpoint(trained_model, args.save_dir, class_to_idx, args.arch, args.layer_1_hidden_units, args.layer_2_hidden_units, args.learning_rate, args.epochs, optimizer)

if __name__ == "__main__":
    main()