import torch
from torchvision import models
import argparse
import helper

def main():
    
    #Define the Command line arguments
    
    parser = argparse.ArgumentParser(description="Predict the class of a flower")
    
    parser.add_argument('image_dir', type=str, help="the path to image to be predictes")
    parser.add_argument('checkpoint_dir', type=str, 
                        help="the path to model checkpoint")
    parser.add_argument('--top_k', type=int, default=5, 
                        help=" the top K predictions")
    parser.add_argument('--category_names', type=str, help="file for mapping categories")
    parser.add_argument('--gpu', action='store_true', help="use gpu if available")
    
    args = parser.parse_args()
    
    model = helper.load_checkpoint(args.checkpoint_dir)
    print('done loading')
    
    img = helper.process_image(args.image_dir)
    
    probs, classes = helper.predict(args.image_dir, model, args.gpu, args.top_k)
    
    if args.category_names:
        cat_to_name = helper.load_category_names(args.category_names)
        class_names = [cat_to_name[i] for i in classes]
    else:
        class_names = classes
        
    
    for prob, class_name in zip(probs[0], class_names):
        prob = prob.item()
        print(f'{class_name}: {prob*100:.2f}%')
        
if __name__ =='__main__':
    main()
     