import os
import shutil
import random
import xmltodict
import yaml
import argparse
import zipfile
from pathlib import Path


# Function to clean the directory except the zip file so as to avoid data issues  
def clean_directory(src_dir):
    for item in src_dir.iterdir():
        if item.name != 'archive.zip':
            if item.is_file():      
                item.unlink()               # Deletes the file
            elif item.is_dir():
                shutil.rmtree(item)
    print(f"Cleaned directory {src_dir}, keeping only archive.zip")
    
    
# Function to unzip the downloaded file
def unzip_archive(src_dir):
    zip_path = src_dir / 'archive.zip'
    if not zip_path.exists():
        raise FileNotFoundError(f"archive.zip not found in {src_dir}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref: # Open the zip file in read mode
        zip_ref.extractall(src_dir)
    
    print(f"Extracted archive.zip in {src_dir}")


# Convert the pascal voc format to yolo format
def convert_xml_to_yolo(xml_path, image_dir, label_dir):
    with open(xml_path) as f:
        data = xmltodict.parse(f.read())        # The parsed data dictionary mirrors the hierarchy of the original XML file
    
    # Extract the filename, width and height information of an image
    # image_name = data['annotation']['filename']
    image_width = int(data['annotation']['size']['width'])
    image_height = int(data['annotation']['size']['height'])

    objects = data['annotation'].get('object', [])      # Get the 'object'(in this case tomato) field from the 'annotation' dictionary and if no object make an empty list
    if not isinstance(objects, list):
        objects = [objects]                # If not a list, make it a list

    yolo_annotations = []
    for obj in objects:
        bbox = obj['bndbox']
        xmin, ymin, xmax, ymax = map(int, [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']])
        # creates a list of the coordinates of the bounding box (minimum x, minimum y, maximum x, maximum y). and convert that to int
       
       # Get the yolo normalized values 
        x_center = (xmin + xmax) / (2 * image_width)
        y_center = (ymin + ymax) / (2 * image_height)
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        
        # Class index set to 0 since there is only tomato
        yolo_annotations.append(f"0 {x_center} {y_center} {width} {height}")    # yolo format <object-class> <x-center> <y-center> <width> <height>
    
    # Creates the image.txt under the label path and then writes the yolo annotations into it
    label_path = Path(label_dir) / f"{Path(xml_path).stem}.txt"
    label_path.write_text("\n".join(yolo_annotations))

def create_ultralytics_structure(src_dir):
    src_dir = Path(src_dir)
    
    # Clean the directory
    clean_directory(src_dir)

    # Unzip the archive
    unzip_archive(src_dir)
    
    # Create new directory structure
    for folder in ['images/train', 'images/val', 'images/test', 'images/predict', 'labels/train', 'labels/val', 'labels/test']:
        (src_dir / folder).mkdir(parents=True, exist_ok=True)

    # Assuming src_dir now contains 'images' and 'annotations' subfolders after unzipping
    image_dir = src_dir / 'images'
    annotation_dir = src_dir / 'annotations'
    
    # Convert XML to YOLO format and prepare file lists
    image_files = list(image_dir.glob('*.png'))     # creates a list of all PNG image files in the image_dir directory as glob('*.png') returns an iterator to all the .png files 
    for xml_file in annotation_dir.glob('*.xml'):
        convert_xml_to_yolo(xml_file, image_dir, src_dir / 'labels/train')

    # Split data
    random.shuffle(image_files)     # randomly reorders all the elements in the image_files list
    train_size = int(0.8 * len(image_files))
    val_size = int(0.15 * len(image_files))
    
    train_files, val_files, test_files = image_files[:train_size], image_files[train_size:train_size+val_size], image_files[train_size+val_size:]
    #Splits the image_files list into three parts based on these sizes

    print(f"Total images: {len(image_files)}, Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    # Move files to their respective directories
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:     #loop iterates over each split ('train', 'val', 'test') and its corresponding file list.
        for file in files:
            dest_img = src_dir / f'images/{split}' / file.name          #For each file, construct the destination paths for both the image and its corresponding label file
            dest_label = src_dir / f'labels/{split}' / f"{file.stem}.txt"
            
            shutil.move(file, dest_img)         # Moves the image file to its new location in the appropriate split directory.
            
            
            # All the label files are initially also stored in the train dir and then moved to the corresponding label directory
            label_file = src_dir / 'labels/train' / f"{file.stem}.txt"
            
            if label_file.exists():
                shutil.move(label_file, dest_label)
            else:
                print(f"Warning: Label file not found: {label_file}")



    # Create data.yaml file
    yaml_content = {
        'train': str(src_dir / 'images/train'),
        'val': str(src_dir / 'images/val'),
        'test': str(src_dir / 'images/test'),
        'nc': 1,
        'names': ['tomato']
    }

    with open(src_dir / 'data.yaml', 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print("Dataset reorganized and data.yaml file created successfully.")

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create Ultralytics dataset structure from XML annotations')
    parser.add_argument('--src_dir', type=str, required=True, help='Path to the source directory containing archive.zip')

    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    create_ultralytics_structure(args.src_dir)