import torch
import cv2
import numpy as np
from torchvision import transforms
import albumentations
from PIL import Image
import cv2
from omegaconf import OmegaConf

from license_plates_text.utils.model_decoders import decode_predictions, decode_padded_predictions
from license_plates_text.model.model import CRNN

classes = ['∅','1','0','4','2','3','6','5','9','7','8','C' ,'A' ,'B', 'H', 'E', 'M', 'K',
 'O', 'T', 'P', 'Y', 'X', 'У']

class LicensePlateText():
    def __init__(self, cfg):
        """
        Initializes the LicensePlateText class with a pre-trained CRNN model.

        :param model_path: Path to the saved model state dictionary (.pth file).
        """
        use_cuda = torch.cuda.is_available() and cfg.basic.use_cuda
        self.device = torch.device(0 if use_cuda else "cpu") #Setup Device

        model_path = cfg.basic.text_model

        self.model = CRNN(dims=256, num_chars=23, use_attention=True, use_ctc=True, grayscale=False)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

    def preprocess_image(self, image: np.ndarray):
        """
        Preprocesses an input image for the CRNN model.

        :param image: Input image as a NumPy array.
        :return: Preprocessed image tensor.
        """
        image = cv2.resize(image, dsize=(180, 50))
        image = np.array(image)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        aug = albumentations.Compose([albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)])

        image = aug(image=image)["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = image[None, ...]
        image = torch.from_numpy(image)
        return image

    def detect_text_in_image(self ,np_image: np.ndarray) -> str:
        """
        Detects text in a given image (provided as a NumPy array).

        :param np_image: Image as a NumPy array (e.g., from cv2.imread or similar).
        :return: Detected text as a string.
        """
        # Convert the image to grayscale if it's not already

        image_tensor = self.preprocess_image(np_image)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            preds, _ = self.model(image_tensor)

        if self.model.use_ctc:
            detected_text = decode_predictions(preds, classes)
        else:
            detected_text = decode_padded_predictions(preds, classes)
        detected_text = ' '.join(detected_text).replace('∅','').split()
        detected_text = ''.join(str(x) for x in detected_text)
        return detected_text
