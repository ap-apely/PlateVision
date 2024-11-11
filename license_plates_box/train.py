import argparse
from ultralytics import YOLO

def train_yolo_model(dataset_path, train_model_dest="./models/yolo_model.pt", epochs=30, imgsz=640, device=0):
    """
    Train and export a YOLO model.
    
    Arguments:
    - dataset_path: Path to the dataset configuration YAML file (default: './datasets/data.yaml')
    - train_model_dest: Path where the trained model will be saved (default: './models/yolo_model.pt')
    - epochs: Number of epochs for training (default: 30)
    - imgsz: Image size for training (default: 640)
    - device: CUDA device (default: 0)
    """
    # Load YOLOv11 model
    model = YOLO("yolo11n.pt")

    # Train the model
    train_results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
    )
    
    # Validate the model
    metrics = model.val()

    # Export the trained model to ONNX format
    path = model.export(format="onnx", save=True, weights=train_model_dest)
    return path

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train and export a YOLOv11 model")
    parser.add_argument('--dataset_path', type=str, default='./datasets/data.yaml', help="Path to dataset YAML file")
    parser.add_argument('--train_model_dest', type=str, default='./models/yolo_model.pt', help="Path to save the trained model")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs for training")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for training")
    parser.add_argument('--device', type=int, default=0, help="CUDA device (default: 0)")

    # Parse arguments
    args = parser.parse_args()

    # Call the training function with parsed arguments
    trained_model_path = train_yolo_model(
        dataset_path=args.dataset_path,
        train_model_dest=args.train_model_dest,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device
    )
    
    # Print the path to the trained model
    print(f"Model exported to: {trained_model_path}")
