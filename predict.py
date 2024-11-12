import argparse
import json
import os
import sys
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from collections import OrderedDict

def get_input_args():
    # define command line arguments
    parser = argparse.ArgumentParser(description='Predict flower name from an image using a trained deep learning model.')

    #get all needed arguments to predict
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint.')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes (default: 5).')
    parser.add_argument('--category_names', type=str, default=None, help='Path to JSON file mapping categories to real names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available.')
    return parser.parse_args()



def load_checkpoint(checkpoint_path):
    #better to catch the error
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' does not exist.")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    arch = checkpoint.get('arch')
    layer_1 = checkpoint.get('layer_1_hidden_units')
    layer_2 = checkpoint.get('layer_2_hidden_units')
    #state_dict = checkpoint.get('state_dict')
    #class_to_idx = checkpoint.get('class_to_idx')

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

    # Freeze parameters as usual
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier
    #same as in train
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, layer_1)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(layer_1, layer_2)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(layer_2, 102)),  
        ('output', nn.LogSoftmax(dim=1))
    ]))

    
    if arch in ['vgg13', 'vgg16']:
        model.classifier = classifier
    elif arch == 'resnet18':
        model.fc = classifier 

    # Load the state dictionary
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image_path):
    #this is similar to part1

    pil_image = Image.open(image_path).convert("RGB")

    # Resize the image where the shortest side is 256 pixels, keeping the aspect ratio
    pil_image.thumbnail((256, 256))  # This ensures the shorter side becomes 256

    # Center crop the image to 224x224
    left = (pil_image.width - 224) / 2
    top = (pil_image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    pil_image = pil_image.crop((left, top, right, bottom))

    # Convert the image to a Numpy array and scale the pixel values from 0-255 to 0-1
    np_image = np.array(pil_image) / 255.0

    # Normalize the image 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std  # subtracting mean and dividing by std

    # Reorder dimensions so that color channel is first (from HWC to CHW)
    np_image = np_image.transpose((2, 0, 1))

    # Convert to torch.Tensor
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)

    return tensor_image

def predict(image_path, model, device, topk=5):
    #i have simply copied it from part 1 and made some minor adjustments

    # Set the model to evaluation mode
    model.eval()

    # Determine the device (GPU if available, else CPU)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Process the image
    np_image = process_image(image_path)

    # Convert the Numpy array to a PyTorch tensor
    image_tensor = torch.from_numpy(np_image).type(torch.FloatTensor)

    # Add a batch dimension (models expect batches of images)
    image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, 3, 224, 224]

    # Move the tensor to the appropriate device
    image_tensor = image_tensor.to(device)

    # Perform forward pass and calculate probabilities
    with torch.no_grad():
        output = model(image_tensor)  # Log probabilities
        probabilities = torch.exp(output)  # Convert to probabilities

    # Get the top K probabilities and their corresponding indices
    top_probs, top_indices = probabilities.topk(topk, dim=1)

    # Convert tensors to lists
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    # Invert the class_to_idx dictionary to get a mapping from index to class
    if hasattr(model, 'class_to_idx'):
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    else:
        raise AttributeError("The model does not have a 'class_to_idx' attribute.")

    # Map the top indices to class labels
    top_classes = [idx_to_class[idx] for idx in top_indices]

    return top_probs, top_classes
   

def load_category_names(category_names_path):
    if not os.path.exists(category_names_path):
        print(f"Error: Category names file '{category_names_path}' does not exist.")
        sys.exit(1)

    with open(category_names_path, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    if args.gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but not available. Using CPU instead.")

    cat_to_name = None
    if args.category_names:
        cat_to_name = load_category_names(args.category_names)


    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.image_path, model, device, topk=args.top_k)


    if cat_to_name:
        classes = [cat_to_name.get(cls, cls) for cls in classes]

    # Print the results!!
    for i in range(len(probs)):
        print(f"{i+1}: {classes[i]} with probability {probs[i]:.4f}")

if __name__ == "__main__":
    main()

