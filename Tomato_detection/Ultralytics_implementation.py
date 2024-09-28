import os
import multiprocessing
import torch
from ultralytics import YOLO
from datetime import datetime
import shutil
import argparse
import gc

# Clean the output project folder if it already exists
def clean_project_folder(project_path):
    if os.path.exists(project_path):
        shutil.rmtree(project_path)
    os.makedirs(project_path)

def main(data_yaml_path, predict_path=None):
    
    # Check if cuda is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")
    
    # Project and run name
    project_name = 'Tomato_detection'
    run_name = f'yolov3_tomato_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    # run_name = f'yolov8_tomato_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # Clean project folder
    project_path = os.path.join(os.getcwd(), project_name)
    # clean_project_folder(project_path)
    

    # Load pre trained yolov3 model
    model = YOLO('yolov3-tiny.pt')          # Using a smaller version of yolo3 model - can be changed to other YOLOv3 variants
    # model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model, can be changed to other YOLOv8 variants

    # Print model information
    print("\nModel Information:")
    model.info()
    
    # Define augmentation parameters
    augment_params = {
    'hsv_h': 0.015,  # HSV-Hue augmentation
    'hsv_s': 0.7,    # HSV-Saturation augmentation
    'hsv_v': 0.4,    # HSV-Value augmentation
    'degrees': 0.0,  # Rotation (degrees)
    'translate': 0.1,  # Translation (+/- fraction)
    'scale': 0.5,    # Scale (+/- gain)
    'shear': 0.0,    # Shear (degrees)
    'flipud': 0.0,   # Flip up-down (probability)
    'fliplr': 0.5,   # Flip left-right (probability)
    'mosaic': 1.0,   # Mosaic augmentation (probability)
    'mixup': 0.0,    # Mixup augmentation (probability)
    }
        
    # Train the model
    results = model.train(data = data_yaml_path, 
                        val = True, 
                        # split = 'val',    # By default does validation on val split
                        epochs = 50, # number of epochs
                        imgsz = 416,  # image size
                        device = device, 
                        seed = 42,
                        batch = 16,    # batch size
                        save = True,  # save results
                        plots = True,
                        cache = False, # disables caching of dataset images in memory. Decreases training speed by increasing disk I/O at the cost of reduced memory usage.
                        project = project_name, # project name
                        name = run_name,  # experiment name
                        workers = 4,   # reduce number of worker threads
                        exist_ok=True,  # Overwrite existing folder
                        augment=True,  # Enable built-in augmentations
                        **augment_params  # Apply custom augmentation parameters
                        
    )
    
    gc.collect()        # Garbage collection - forces to reclaim memory that is no longer in use by the program.
    torch.cuda.empty_cache()    # Releases all unoccupied cached memory currently held by the caching allocator so that those can be used in other GPU operations.
      
    # Evaluate the model on the test set
    val_results = model.val(data = data_yaml_path, 
                            split = 'test',
                            imgsz = 416,  # image size
                            batch = 16,    # batch size
                            half = True,    # Enables half-precision (FP16) computation, reducing memory usage and potentially increasing speed with minimal impact on accuracy
                            device = device, 
                            plots = True,
                            save_json=True                            
                            )

    print("Validation Results:")
    print(f"mAP50-95 for validation set: {val_results.box.map}")
    print(f"mAP50 for validation set: {val_results.box.map50}")
    print(f"mAP75 for validation set: {val_results.box.map75}")
    print(f"Precision for validation set: {val_results.box.mp}")
    print(f"Recall for validation set: {val_results.box.mr}")
    
    results_path = os.path.join(project_path, run_name, 'validation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"mAP50-95: {val_results.box.map}\n")
        f.write(f"mAP50: {val_results.box.map50}\n")
        f.write(f"mAP75: {val_results.box.map75}\n")
        f.write(f"Precision: {val_results.box.mp}\n")
        f.write(f"Recall: {val_results.box.mr}\n")
    
    
    if predict_path:
               
        print("\n Predcitions on unseen test images")
        # Perform inference on the unseen data
        predict_results = model.predict(source = predict_path, 
                                    save = True,     # Enables saving of the annotated images to file
                                    save_txt = True,  # Saves detection results in a text file, following the format [class] [x_center] [y_center] [width] [height] [confidence]
                                    save_conf = True, # 	Includes confidence scores in the saved text files
                                    imgsz = 416,  # image size
                                    visualize = False, # Activates visualization of model features during inference, providing insights into what the model is "seeing". Useful for debugging and model interpretation
                                    half = True,    # Enables half-precision (FP16) computation, reducing memory usage and potentially increasing speed with minimal impact on accuracy
                                    augment = True, # Enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed
                                    device = device, 
                                    plots = True )
        
        print("\n Predcited images with the bounding boxes saved.")
        
    
    # Save the trained model
    model_save_path = os.path.join(project_path, run_name, f'{run_name}.pt')
    model.save(model_save_path)

    print(f"Training, validation, and testing completed. Model saved to {model_save_path}")
    

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Needed for Windows, prevents the main module from being executed multiple times when the program is run. had to include since it gave issues on my system.
        
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train and evaluate YOLO model')
    parser.add_argument('--data_yaml_path', type=str, required=True, 
                        help='Path to the data.yaml file')
    parser.add_argument('--predict_path', type=str, default=None,
                    help='Path to images for prediction after training (optional)')
                            
    # Parse arguments
    args = parser.parse_args()
    
    # Run main function with provided paths
    main(args.data_yaml_path, args.predict_path)