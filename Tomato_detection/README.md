# 1. Tomato Detection

The Project [Tomato Detection dataset](https://www.kaggle.com/datasets/andrewmvd/tomato-detection/data) from Kaggle focussed on Object detection.

## Documentation

### Prerequisites

Make sure that the python libraries mentioned in the requirements.txt file have been installed in your python environment. For example in a conda environment, you can run
* conda
  ```sh
  conda install pandas=2.2.2
  ```

### Dataset Directory Structure

_Many object detection models like YOLO require a specific directory structure for storing the data which as shown below._

1. Clone the repo
   ```sh
   git clone [https://github.com/github_username/repo_name.git](https://github.com/JarnoRFB/ml-engineering-task-bhaskar.git)
   ```
2. Download the zipped data from [Tomato Detection dataset](https://www.kaggle.com/datasets/andrewmvd/tomato-detection/data) into an empty folder where the required directory structure will be created. We can copy this zipped file to the data folder in the cloned repo
   
3. Run the data_structure_src.py script using the following command
   ```sh
   python .\data_structure_src.py --src_dir '\path\to\zipped\file'
   ```
This expects --src_dir argument which is the path where the zipped file is present and creates the following data structure. Everytime the code is run, it first cleans the folder except the zipped file and then goes on to create the data structure.  

```dataset_root/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── test/  
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── predict/  (for inference on new images)
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   ├── val/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   └── test/  
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
└── data.yaml
```

### Training the model

Execute the Ultralytics_implementation.py script which trains the YOLO model using the popular Ultralytics library for object detection. <br />

```sh
python .\Ultralytics_implementation.py --data_yaml_path 'path\to\data.yaml' --predict_path 'path\to\predict'
```
   - It expects the --data_yaml_path argument which points to the data.yaml file created in the previous step and provides the model with the train, val and test images, labels and class information. <br />
   - The optional --predict_path argument computes predictions for new data. The new unseen images need to be copied to the predict folder that's created in the previous data directory and the path is passed in this argument.
<img src="https://github.com/JarnoRFB/ml-engineering-task-bhaskar/blob/development/Tomato_detection/yolov3_tomato_20240827_223827/tomato642.png" width="416" height="352"> 



This script will first download the pretrained YOLO model, do the training and then store the time stamped model outputs for every run inside the folder ./Tomato_detection/

### Model results
Inside every model output folder stored in ./Tomato_detection/, validation_results.txt stores the model performance on the test dataset. The folder also includes more comprehensive information on the model like the recall and precision curves, training and validation loss plots across different metrics, a csv file with a log of changes over the epochs etc

In this case study, we choose the mean Average Precision(mAP) metric to measure the performance of the model which is based on the precision and recall values at different IoU thresholds. The current model has
- a Precision of 85.4% which means that of all the predictions we made, around 85% were correct predictions
- a Recall of 73.4% which means that we correctly identified nearly 73.4% of tomatoes
- mAP50 of 85.14% which refers to mAP at 50% Intersection over Union(IOU) threshold

mAP50 is preferred over mAP75 and mAP90 in this experiment because

- Inherently difficult to localize precisely: in our data a lot of tomatoes are hidden behind leaves or occluded by other tomatoes
- Exact localization is less critical: we prioritize detecting a tomato more than precisely localizing it

<img src="https://github.com/JarnoRFB/ml-engineering-task-bhaskar/blob/development/Tomato_detection/yolov3_tomato_20240827_223827/results.png" width="800" height="500">

### Model Improvements
Some of the ways we can further increase this accuracy:

- Increase training time: increase the number of epochs since we see from loss curve that it still hasn't plateaued
- Adjust the image size: try larger image sizes (e.g., 640 or 832) instead of 416
- Data augmentation: experiment with different augmentation parameters and also consider adding more augmentations like random crop, blur, or noise
- Use a larger model instead of yolov3-tiny.pt
- Data cleaning: ensure the robustness and correctness of the labeled dataset
- Hyperparameter tuning: using techniques like grid search or Bayesian optimization to find optimal hyperparameters
