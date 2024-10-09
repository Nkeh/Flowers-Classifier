import torch
import argparse
from torchvision import models
import helper
import time

def main():
    #defining the command line arguments
    parser = argparse.ArgumentParser(description='Predicts the name of a flower')
    
    parser.add_argument('data_dir', type=str, help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='', 
                        help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='densenet121', 
                        help='architecture for trainig')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='The learning rate used to adjust weights')
    parser.add_argument('--epochs', type=int, default=5, 
                        help='# of times the model iterates through dataset when training')
    parser.add_argument('--hidden_units', type=int, default=5, 
                        help='Number of hidden units')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args=parser.parse_args()
    
    train_loader, valid_loader, class_to_idx = helper.load_data(args.data_dir)
    
    model, optimizer, criterion = helper.build_model(args.arch, args.hidden_units,
                                                     args.learning_rate)
    
    training_time = time.time()
    helper.train_model(model, train_loader, valid_loader, optimizer, criterion, args.epochs, args.gpu)
    
    helper.save_model(model, optimizer, class_to_idx, args.save_dir, args.arch, args.epochs, training_time)
    
if __name__ == '__main__':
    main()