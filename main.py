import os
import torch
from ExtendedLocalisation import ExtendedLocalisation

# MLP Handwriting Recognition and Localization
# Author: Muhamad Faez Abdullah

# Best to Date:
# Results: Test Loss: 1.4532, Test Accuracy: 0.6129, Test Precision: 0.6619, Test Recall: 0.6129, Test F1 Score: 0.6092
# Version Used: DatasetProcessorV4 and ModelV13 and TrainerV4(Patience = 20) w Learning Rate = 0.0001

#DatasetProcessor Version Control:
#from DatasetProcessor import DatasetProcessor #DatasetProcessor V1: Output size 64*64 with No Data Augmentation (Initial Model)
#from DatasetProcessorV2 import DatasetProcessorV2 #DatasetProcessor V2: Output size 64*64 with Scaling as Data Augmentation (Improvement Seen) 
#from DatasetProcessorV3 import DatasetProcessorV3 #DatasetProcessor V3: Incrase Data Augmentation ontop of Scaling (Decrease in Model's Accuray and Increase in Test Loss)
from DatasetProcessorV4 import DatasetProcessorV4 #DatasetProcessor V4: Only with Scaling but Increase Output size to 128*128 (Improvement Seen)
#from DatasetProcessorV5 import DatasetProcessorV5 #DatasetProcessor V5: Increase in Output Size from 128*128 to 256*256 (Similar results with a drastic increase in computation power. Hence will not be used)
#from DatasetProcessorV6 import DatasetProcessorV6 #DatasetProcessor V6: Reverted Back to V4 but increase batch size to 128 (Decrease in Model's Accuracy and Increase in Test Loss)
#from DatasetProcessorV7 import DatasetProcessorV7 #DatasetProcessor V7: Similar to V4 but Decreased Batch Size to 32 (Decrease in Model's Accuray and Increase in Test Loss)

# HandWritingMLPModel Version Control:
#from HandWritingMLPModelV1 import HandWritingMLPModelV1 # HandWritingMLPModel Version 1 : Initial Model based of 64*64 output size 
#from HandWritingMLPModelV2 import HandWritingMLPModelV2 # HandWritingMLPModel Version 2 : Increased Layer Complexity and Implemented Dropout of 0.5 
#from HandWritingMLPModelV3 import HandWritingMLPModelV3 # HandWritingMLPModel Version 3 : Converted Function to Leaky Relu 
#from HandWritingMLPModelV4 import HandWritingMLPModelV4 # HandWritingMLPModel Version 4 : Duplicate of Version3 w Trainer V2(Increased Patience Level) w DataSetProcessor V2(Scaling Augmentation)
#from HandWritingMLPModelV5 import HandWritingMLPModelV5 # HandWritingMLPModel Version 5 : Decreased Drop Out Rate from V4 
#from HandWritingMLPModelV6 import HandWritingMLPModelV6 # HandWritingMLPModel Version 6 : Remove Dropout rate to see if further drop will lead to an increase in test accuray and decrease in test loss
#from HandWritingMLPModelV7 import HandWritingMLPModelV7 # HandWritingMLPModel Version 7 : Editted it to suit the new Output Size of 128*128
#from HandWritingMLPModelV8 import HandWritingMLPModelV8 # HandWritingMLPModel Version 8: Editted to suit the new Output Size of 256*256
#from HandWritingMLPModelV9 import HandWritingMLPModelV9 # HandWritingMLPModel Version 9: Decrease Layer Width by Half while maintaning overall architecture the same
#from HandWritingMLPModelV10 import HandWritingMLPModelV10 #HandWritingMLPModel Version 10: Increase Layer Width * 2 while maintaining overall architecture the same
#from HandWritingMLPModelV11 import HandWritingMLPModelV11 #HandWritingMLPModel Version 11: Added Layers to Original Architecture for smoothing
#from HandWritingMLPModelV12 import HandWritingMLPModelV12 #HandWritingMLPModel Version 12: Added Further Layers to smooth out transition
from HandWritingMLPModelV13 import HandWritingMLPModelV13 #HandWritingMLPModel Version 13: Added One Layer of 2048 on top.
#from HandWritingMLPModelV14 import HandWritingMLPModelV14 #HandWritingMLPModel Version 14: Smooth Layer Transition between 2048 and 1024 (No Improvement)
#from HandWritingMLPModelV15 import HandWritingMLPModelV15 #HandWritingMLPModel Version 15: Revert back to V13 but added one layer ontop (4098) (No Improvement)
#from HandWritingMLPModelV16 import HandWritingMLPModelV16 #HandWritingMLPModel Version 16: Revert back to 13 but change function to Relu
#from HandWritingMLPModelV17 import HandWritingMLPModelV17 #HandWritingMLPModel Version 17: Change Function to ELU
#from HandWritingMLPModelV18 import HandWritingMLPModelV18 #HandWritingMLPModel Version 18: Change Functions to Selu

# HandWritingMLPModelTrainer Version Control:
#from HandWritingMLPModelTrainerV1 import HandWritingMLPModelTrainerV1 #HandWritingMLPModel V1: First Version with a Patience of 5
#from HandWritingMLPModelTrainerV2 import HandWritingMLPModelTrainerV2 #HandWritingMLPModel V2: Increased Patience to 10
#from HandWritingMLPModelTrainerV3 import HandWritingMLPModelTrainerV3 #HandWritingMLPModel V3: Change Optimiser to SGD
from HandWritingMLPModelTrainerV4 import HandWritingMLPModelTrainerV4 #Reverted Back to Old Optimiser and Increased Patience to 20
#from HandWritingMLPModelTrainerV5 import HandWritingMLPModelTrainerV5 #Increase Patience even higher to 50


def main():
    # Determine if a GPU is available and if not, use a CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    # Get the directory in which the main.py script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Construct the path to the CSV file and image folder relative to the script's location
    csv_file_path = os.path.join(script_dir, 'archive', 'english.csv')
    img_folder_path = os.path.join(script_dir, 'archive', 'Img')
    
    # Initialize the DatasetProcessor with the constructed paths
    #processor = DatasetProcessor(csv_file_path=csv_file_path, img_folder_path=img_folder_path)
    #processor = DatasetProcessorV2(csv_file_path=csv_file_path, img_folder_path=img_folder_path)
    #processor = DatasetProcessorV3(csv_file_path=csv_file_path, img_folder_path=img_folder_path)
    processor = DatasetProcessorV4(csv_file_path=csv_file_path, img_folder_path=img_folder_path)
    #processor = DatasetProcessorV5(csv_file_path=csv_file_path, img_folder_path=img_folder_path)
    #processor = DatasetProcessorV6(csv_file_path=csv_file_path, img_folder_path=img_folder_path)
    #processor = DatasetProcessorV7(csv_file_path=csv_file_path, img_folder_path=img_folder_path)

    # Use the DatasetProcessor to load and split the dataset
    train_loader, val_loader, test_loader = processor.load_and_split_dataset()
    print("Data loading complete. DataLoaders are ready to be used.")

    # Initialize the model
    #model = HandWritingMLPModelV1().to(device) # HandWritingMLPModel Version 1 
    #model = HandWritingMLPModelV2().to(device) # HandWritingMLPModel Version 2
    #model = HandWritingMLPModelV3().to(device) # HandWritingMLPModel Version 3  
    #model = HandWritingMLPModelV4().to(device) # HandWritingMLPModel Version 4 
    #model = HandWritingMLPModelV5().to(device) # HandWritingMLPModel Version 5
    #model = HandWritingMLPModelV6().to(device) # HandWritingMLPModel Version 6
    #model = HandWritingMLPModelV7().to(device) # HandWritingMLPModel Version 7
    #model = HandWritingMLPModelV8().to(device) # HandWritingMLPModel Version 8
    #model = HandWritingMLPModelV9().to(device) # HandWritingMLPModel Version 9
    #model = HandWritingMLPModelV10().to(device) # HandWritingMLPModel Version 10
    #model = HandWritingMLPModelV11().to(device) # HandWritingMLPModel Version 11
    #model = HandWritingMLPModelV12().to(device) # HandWritingMLPModel Version 12
    model = HandWritingMLPModelV13().to(device) # HandWritingMLPModel Version 13
    #model = HandWritingMLPModelV14().to(device) # HandWritingMLPModel Version 14
    #model = HandWritingMLPModelV15().to(device) # HandWritingMLPModel Version 15
    #model = HandWritingMLPModelV16().to(device) # HandWritingMLPModel Version 16
    #model = HandWritingMLPModelV17().to(device) # HandWritingMLPModel Version 17
    #model = HandWritingMLPModelV18().to(device) # HandWritingMLPModel Version 18

    # Initialize the trainer with the model, data loaders (including test_loader), and training parameters
    #trainer = HandWritingMLPModelTrainerV1(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, epochs=1000, lr=0.001, device=device)
    #trainer = HandWritingMLPModelTrainerV2(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, epochs=1000, lr=0.0001, device=device)
    #trainer = HandWritingMLPModelTrainerV3(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, epochs=1000, lr=0.0001, device=device)
    trainer = HandWritingMLPModelTrainerV4(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, epochs=1000, lr=0.0001, device=device)
    #trainer = HandWritingMLPModelTrainerV5(model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, epochs=1000, lr=0.0001, device=device)


    # Start training and validation
    print("Starting training process...")
    trainer.train()
    print("Training completed. Validation and Testing metrics are logged in TensorBoard.")
    
    # Construct the path to the Hello.png file relative to the script's location
    hello_img_path = os.path.join(script_dir, 'Hello.jpg')

    model_path = os.path.join(script_dir, 'best_model.pth')

    # Initialize the ExtendedLocalisation class and perform localization
    print("Starting character localization on Hello.jpg...")
    localizer = ExtendedLocalisation(model_path)
    predictions = localizer.localize_characters(hello_img_path)

     # Optionally, print or process the predictions
    for x, y, character, confidence in predictions:
        print(f"Character '{character}' detected at ({x}, {y}) with confidence {confidence:.2f}")


if __name__ == '__main__':
    main()
