import argparse
import cv2
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

import hydra

from rich.console import Console
from rich import print
from rich.table import Table

from license_plates_box.license_plates import LicensePlateDetector, Detection
from license_plates_text.license_plates_text import LicensePlateText


console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="License Plate Detection and Text Recognition")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    return parser.parse_args()

@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg):
    args = parse_args()

    image_path = args.image_path

    console.print("[bold blue]Initializing models...[/bold blue]")
    license_plate_detector = LicensePlateDetector(cfg=cfg)
    license_plate_reader = LicensePlateText(cfg=cfg)

    console.print(f"[bold green]Loading image from:[/bold green] {image_path}")
    image = cv2.imread(image_path)
    original_shape = image.shape
    console.log(f"Original image shape: {original_shape}")

    image = cv2.resize(
        image,
        (cfg.additional.image_resolution_x, cfg.additional.image_resolution_y),
        fx=0,
        fy=0,
        interpolation=cv2.INTER_CUBIC
    )
    resized_shape = image.shape
    console.log(f"Resized image shape: {resized_shape}")

    table = Table(title="License Plate Detection and Text Recognition Results")
    table.add_column("License Plate #", justify="center", style="cyan")
    table.add_column("Original Shape", justify="center", style="magenta")
    table.add_column("Resized Shape", justify="center", style="magenta")
    table.add_column("Detected Text", justify="center", style="green")

    # Detect license plates
    console.print("[bold yellow]Detecting license plates...[/bold yellow]")
    license_plates = license_plate_detector.detect_license_plate(image)
    console.log(f"Number of license plates detected: {len(license_plates)}")

    for i, license_plate in enumerate(license_plates, 1):
        # Crop and process each detected license plate
        console.print(f"[bold cyan]Processing license plate {i}...[/bold cyan]")
        license_plate_crop = image[
            int(license_plate.y1):int(license_plate.y2),
            int(license_plate.x1):int(license_plate.x2),
            :
        ]
        console.log(f"License plate {i} crop shape: {license_plate_crop.shape}")

        # Detect text in the cropped license plate
        console.print(f"[bold magenta]Reading text for license plate {i}...[/bold magenta]")
        license_plate.text = license_plate_reader.detect_text_in_image(license_plate_crop)
        console.log(f"Detected text for license plate {i}: {license_plate.text}")

        # Output detected text
        console.print(f"[bold white on green]License Plate {i} Text:[/bold white on green] {license_plate.text}")

        table.add_row(
            str(i),
            str(original_shape),
            str(resized_shape),
            license_plate.text
        )

        image = cv2.rectangle(image, (license_plate.x1, license_plate.y1), (license_plate.x2, license_plate.y2), (0, 255, 0), 2)  # Green box
        cv2.putText(image, license_plate.text, (license_plate.x1, license_plate.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    console.print(table)
    cv2.imshow("License Plate Detection", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()