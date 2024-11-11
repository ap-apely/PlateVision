# PlateVision 🚗✨

A computer vision system designed to detect license plates and recognize text within them.

## Table of Contents 📚

- [About](#about)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Config](#config)
- [Traing](#training)

## About 🔍

**PlateVision** 🚗✨ is an advanced computer vision system designed to detect license plates and recognize the alphanumeric text within them. The system utilizes state-of-the-art deep learning techniques, specifically the **YOLOv11** model for object detection 🧠 and a **Convolutional Recurrent Neural Network (CRNN)** for optical character recognition (OCR) 📜. PlateVision is built to be efficient, flexible, and highly accurate in real-world applications, such as vehicle surveillance 🚓, toll collection 💸, and automated parking systems 🅿️.

The system is composed of two core components:

1. **License Plate Detector** 🚙🔍:  
   The detection module uses **YOLOv11**, a cutting-edge object detection model, to identify license plates in an image or video frame 🎥. YOLOv11 is a fast and accurate deep learning model capable of detecting multiple objects in real-time ⏱️. It is well-suited for high-speed applications like vehicle monitoring. This model processes images to predict bounding boxes around license plates, enabling the system to focus on relevant areas for text recognition.

2. **License Plate Text Recognition** 📝🔠:  
   After detecting the license plate, the system uses a **CRNN** model to recognize the alphanumeric text within the plate 🅾️🔡. A CRNN combines the power of **Convolutional Neural Networks (CNNs)** for feature extraction 🔍 and **Recurrent Neural Networks (RNNs)** for sequence modeling 🔁. This makes it highly effective for text recognition tasks, especially when the text is skewed, noisy, or in non-standard fonts. The CRNN model converts the extracted features from the license plate into readable text.
   
### Use Cases 🌍:
- **Vehicle Identification** 🚙🔑: Automating vehicle registration and identification in smart city applications 🏙️, including toll booths 🚏, automated parking 🅿️, and security systems 🛡️.
- **Law Enforcement** 👮‍♂️: Assisting in tracking vehicles of interest 🚔 or enforcing traffic laws 🛣️ by capturing license plate data in surveillance footage 🎥.
- **Fleet Management** 🚛: Automatically tracking and recording the location of fleet vehicles for logistics companies 🚚💨.

PlateVision is built for ease of use, ensuring that both researchers 🧑‍🔬 and developers 👨‍💻 can quickly integrate license plate detection and text recognition into their own projects. With a strong focus on performance ⚡ and accuracy 🎯, it is an ideal solution for applications where automated license plate reading is essential.

## Features 🌟:
- **High Accuracy** 🎯: The combination of YOLOv11 and CRNN ensures reliable and precise detection and recognition of license plates, even in challenging conditions such as varying lighting 🌅, angles ↗️, or occlusions 🛑.
- **Real-Time Processing** ⚡: The system is designed to work efficiently for real-time applications, providing fast and scalable performance 🖥️.
- **Flexible Input** 🎨: Supports both grayscale and color images 🌈 with a variety of resolutions 📸, making it adaptable to various sources like CCTV footage or static images.
- **Scalability** 📈: Easily adaptable for large-scale systems, such as traffic monitoring 🚦 or fleet management 🚚, with the ability to handle batch image processing.

## Requirements 🛠️

To run the project, make sure you have the following:

- Python 3.8 or higher
- PyTorch 1.9 or higher
- OpenCV 4.5 or higher
- Ultralytics (YOLOv11 model)
- Albumentations (image augmentation library)

## Installation ⚙️

Follow these steps to set up the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/ap-apely/PlateVision.git
    cd PlateVision
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. If you encounter any issues, make sure to update your Python version and dependencies.

## Usage 🎬

### License Plate Detection and Text Recognition

Run the main script to detect license plates and recognize text in an image:

```bash
python main.py /path/to/image.jpg
```

## Config ⚙️

The configuration file allows you to easily adjust the settings for the project. Below is an example of a typical configuration:

```yaml
basic:
  use_cuda: true
  yolo_model: ./license_plates_box/weights/license_plates_detection.pt
  text_model: ./license_plates_text/weights/license_plates_text.pth
  
additional:
  image_resolution_x: 1920
  image_resolution_y: 1080

text_recognition:
  defaults:
    - override hydra/job_logging: custom
    
  processing:
    device: cuda
    image_width: 180
    image_height: 50 
    
  training:
    lr: 3e-4
    batch_size: 8
    num_workers: 4
    num_epochs: 10

  bools:
    DISPLAY_ONLY_WRONG_PREDICTIONS: true
    VIEW_INFERENCE_WHILE_TRAINING: true
    SAVE_CHECKPOINTS: false

  paths:
    dataset_dir: ./dataset/train/img
    save_model_as: ./logs/2.pth

  model:
    use_attention: true 
    use_ctc: true
    gray_scale: true
    dims: 256
```

### Explanation 📝:
- **basic**: Contains general settings such as whether to use **CUDA** for GPU acceleration ⚡ and paths to the **YOLO** 🚗 and **CRNN** 🧠 models.
- **additional**: Defines the **image resolution** for input images 📸.
- **text_recognition**: Contains configuration related to **text recognition** 📝, including processing settings (e.g., device, image size), training parameters (e.g., learning rate, batch size), and paths for **dataset** 📂 and **model saving** 💾.
- **model**: Contains model-specific settings like whether to use **attention mechanisms** 👀 or **grayscale input** 🖤.
  
You can adjust these settings based on your system's specifications 🖥️ and the specific needs of your project 🛠️.

## Training 🏋️‍♂️

### Text Recognition Model 📜

#### How to Train 📝:
1. **Prepare the Dataset**:  
   Create a directory called **"dataset"** and place your images in it (preferably **PNG**, but other formats are acceptable as long as you modify the code to handle them). The name of each image must correspond to the text written in the image.

2. **Directory Structure**:
   Your file tree should look like this:
   PlateVision
   └─license_plates_text
     └─Dataset
       ***.png

4. **Image Names**:  
The name of each image should represent the text it contains. For example, `car.png` should have the word "car" written on it, `cat.png` should have "cat", and so on.

5. **Data Preparation**:  
Ensure that all your images are of the same length. Padding is done automatically when using **Attention + CrossEntropy**. However, if you're using **CTC Loss**, padding is not automatically applied, so make sure to normalize your target lengths. You can add a special character to represent empty space, but avoid using the same character as the **blank token** in CTC (they are different). For example, 'car_.png', 'tree.png'

### YOLO Visual Model 🚗

#### How to Train YOLO Model 📸:
1. **Prepare Your Dataset**:
   - Create a **data.yaml** file that specifies the paths to your training and validation data, and define the class labels.

   Example of a `data.yaml` file:
   ```yaml
   train: ../train/images
   val: ../valid/images
   test: ../test/images
    
   nc: 1
   names: ['License_Plate']
   ```

2. **Training**:
   - With the YOLOv11 model, the training is done via the `train.py` script from the **Ultralytics** repository. This script handles data loading, model training, validation, and exporting the trained model.

3. **Run Training**:
   Here’s an example of how you can train the YOLO model using the `train.py` script:
   ```bash
   python train.py
   ```

4. **Model Export**:
   After training, you can export the model to the desired format (e.g., ONNX or TorchScript) for deployment. The exported model can then be used for inference in your vehicle detection system.
## Additional information about the future of this project...
*(P.S I may upgrade this repo in the future by adding my own pre-trained weights and creating a real-time web-based interface. Star the project to support me!)*
