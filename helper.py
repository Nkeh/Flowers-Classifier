import numpy as np

import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import Image

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

def load_data(data_dir):
    train_dir = data_dir+"/train"
    valid_dir = data_dir+"/valid"
    
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                       'valid': transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])
    }
    # Load the datasets with ImageFolder
    Image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']), 
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    data_loaders = {
        'train': torch.utils.data.DataLoader(Image_datasets['train'], batch_size=64, shuffle=True,
                                              drop_last=True),
        'validate': torch.utils.data.DataLoader(Image_datasets['valid'], batch_size=64,
                                                drop_last=True)
    }
    
    return data_loaders['train'], data_loaders['validate'], Image_datasets['train'].class_to_idx




def build_model(arch='densenet121', hidden_units=5, learning_rate=0.001):
    
    hid = hidden_units+2
    if arch=='densenet121':
        model = models.densenet121(pretrained=True)
        hidden_layers_list = list(np.linspace(1024, 102, num=hid).astype(int))[1:-1]
        layers = [nn.Linear(1024, hidden_layers_list[0]), nn.ReLU(), nn.Dropout(p=0.2)]
    elif arch=='vgg16':
        model = models.vgg16(pretrained=True)
        hidden_layers_list = list(np.linspace(4096, 102, num=hid).astype(int))[:-1]
        layers = [nn.Linear(25088, hidden_layers_list[0]), nn.ReLU(), nn.Dropout(p=0.2)]
    else:
        raise ValueError('Unsupported Architecture')
        
        
        
    for i in range(0, len(hidden_layers_list)-1):
        layers.append(nn.Linear(hidden_layers_list[i], hidden_layers_list[i+1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.2))
        
    layers.append(nn.Linear(hidden_layers_list[-1], 102))
    layers.append(nn.LogSoftmax(dim=1))
    
    model.classifier = nn.Sequential(*layers)
    print(model.classifier)
        
#         for param in model.parameters():
#             param.requires_grad = False
            
#         model.classifier = nn.Sequential(nn.Linear(1024, 512),
#                            nn.ReLU(),
#                            nn.Dropout(p=0.2),
#                            nn.Linear(512, 256),
#                            nn.ReLU(),
#                            nn.Dropout(p=0.2),
#                            nn.Linear(256, 102),
#                            nn.LogSoftmax(dim=1)
#                           )
        
#     elif arch=='vgg16':
#         model = models.vgg16(pretrained=True)
#         for param in model.parameters():
#             param.requires_grad = False
        
#         model.classifier = nn.Sequential(nn.Linear(25088, 4096),
#                            nn.ReLU(),
#                            nn.Dropout(p=0.2),
#                            nn.Linear(4096, 1024),
#                            nn.ReLU(),
#                            nn.Dropout(p=0.2),
#                            nn.Linear(1024, 102),
#                            nn.LogSoftmax(dim=1)
#                           )
        

        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, optimizer, criterion


def train_model(model, trainloader, validloader, optimizer, criterion, epochs, gpu):
    steps = 0
    training_loss = 0
    print_every = 10
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    print(f'training model on {device}')
    model.to(device)
    for epoch in range(epochs):
        for inputs, labels in trainloader:

            #Checking the size of input and labels
            if inputs.size(0) != labels.size(0):
                continue

            steps += 1

            #moving inputs and labels to gpu
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)

            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss = loss.item()

            if steps%print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():

                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)

                        batch_loss = criterion(logps, labels)

                        validation_loss = batch_loss.item()

                        #Calculating the accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {training_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")
                training_loss = 0
                model.train()


            
def save_checkpoint(model, optimizer, class_to_idx, save_dir, arch, epochs):
    checkpoint = {'architecture': arch,
                   'input_size': model.classifier[0].in_features,
                   'output_size': 102,
                   'class_to_idx': class_to_idx,
                   'optimizer_state': optimizer.state_dict(),
                   'epochs': epochs,
                   'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir)
    

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model, optimizer, criterion = build_model(checkpoint['architecture'])
    
    model.load_state_dict(checkpoint['state_dict'])
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #Process a PIL image for use in a PyTorch model
    #Opening the image and croping
    pil_image = Image.open(image)
    aspect_ratio = pil_image.size[0]/pil_image.size[1]
    if pil_image.size[0] < pil_image.size[1]:
        new_size = (256, int(256 / aspect_ratio))
    else:
        new_size = (int(256 * aspect_ratio), 256)
    
    pil_image = pil_image.resize(new_size)
    
    #Croping the center 224 by 224
    width, height = pil_image.size
    new_width = new_height = 224
    left = (width - new_width)/2
    right = (width + new_width)/2
    top = (height - new_height)/2
    bottom = (height + new_height)/2
    
    pil_image = pil_image.crop((left, top, right, bottom))
    
    #Convert to Numpy array and scale the pixels from 0-255 to 0-1 
    np_image = np.array(pil_image) / 255.0
    
    #Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.244, 0.255])
    np_image = (np_image - mean) / std
    
    #Reorder the dimensions
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.tensor(np_image).float()

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def load_category_names(json_file_path):
    with open(json_file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #Implement the code to predict the class from an image file
    img = process_image(image_path)
    
    #making sure the tensor is of right type
    img_input = img.unsqueeze_(0).type(torch.FloatTensor)
    
    
    #if model should use gpu or not
    if gpu and torch.cuda.is_available():
        model.to('cuda')
        img_input = img_input.to('cuda')
        
    else:
        model.cpu()
        
    
    model.eval()
    
    with torch.no_grad():
        output = model.forward(img_input)
        
    ps = torch.exp(output)
    
    top_probs, top_indices = ps.topk(topk, dim=1)

    idx_to_class = {v: k for k,v in model.class_to_idx.items()}
    
    top_indices = top_indices.cpu()
    top_classes = [idx_to_class[i.item()] for i in top_indices[0]]
    
    top_probs = top_probs.cpu().numpy()    
    
    return top_probs, top_classes
    
    
    
