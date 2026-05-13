import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset


# RGB PREPROCESSING

class RGBPreprocessor:
    """
    Tiền xử lý ảnh RGB cho CNN branch.

    Pipeline:
        Resize
        Data Augmentation
        ToTensor
        Normalize
    """

    def __init__(self, image_size=224, train=True):

        self.image_size = image_size
        self.train = train

        
        # Data augmentation cho train set
        
        if self.train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),

                # Random flip
                T.RandomHorizontalFlip(p=0.5),

                # Random rotation
                T.RandomRotation(degrees=15),

                # Điều chỉnh sáng/tương phản
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2
                ),

                # Chuyển sang tensor
                T.ToTensor(),

                # Normalize theo chuẩn ImageNet
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        
        # Validation/Test transform
        
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),

                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __call__(self, image):
        """
        Parameters
        ----------
        image : PIL.Image

        Returns
        -------
        tensor : torch.Tensor
            Shape: (3, H, W)
        """

        return self.transform(image)


# FFT PREPROCESSING

class FFTPreprocessor:
    """
    Tiền xử lý ảnh Frequency Domain.

    Pipeline:
        RGB -> Gray
        FFT
        FFT Shift
        Magnitude Spectrum
        Log Transform
        Normalize
        Resize
        To Tensor
    """

    def __init__(self, image_size=224):

        self.image_size = image_size

    def compute_fft(self, image):
        """
        Tính FFT magnitude spectrum.

        Parameters
        ----------
        image : numpy.ndarray
            RGB image

        Returns
        -------
        log_magnitude : numpy.ndarray
            FFT magnitude spectrum
        """

        # Convert sang grayscale      
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        
        # FFT 2D 
        fft = np.fft.fft2(gray)

        
        # Shift zero-frequency component vào giữa ảnh    
        fft_shift = np.fft.fftshift(fft)

        
        # Magnitude spectrum       
        magnitude = np.abs(fft_shift)

        
        # Log transform
        #
        # Giúp giảm dynamic range      
        log_magnitude = np.log(magnitude + 1)

        return log_magnitude

    def normalize(self, image):
        """
        Normalize FFT image về [0,1]
        """

        min_val = image.min()
        max_val = image.max()

        image = (image - min_val) / (max_val - min_val + 1e-8)

        return image

    def __call__(self, image):
        """
        Parameters
        ----------
        image : PIL.Image

        Returns
        -------
        tensor : torch.Tensor
            Shape: (1, H, W)
        """

        
        # PIL -> numpy       
        image = np.array(image)

        
        # FFT processing      
        fft_image = self.compute_fft(image)

        
        # Normalize      
        fft_image = self.normalize(fft_image)

        
        # Resize      
        fft_image = cv2.resize(
            fft_image,
            (self.image_size, self.image_size)
        )

        
        # Convert sang float32      
        fft_image = fft_image.astype(np.float32)

        
        # Add channel dimension
        #
        # (H,W) -> (1,H,W)     
        fft_image = np.expand_dims(fft_image, axis=0)

        
        # Convert sang tensor      
        fft_tensor = torch.from_numpy(fft_image)

        return fft_tensor



# JPEG COMPRESSION AUGMENTATION

class JPEGCompression:
    """
    Giả lập JPEG compression.

    Watermark thường bị ảnh hưởng bởi nén JPEG,
    augmentation này giúp model robust hơn.
    """

    def __init__(self, quality=80):
        self.quality = quality

    def __call__(self, image):

        # PIL -> numpy
        image = np.array(image)

        # RGB -> BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Encode JPEG
        encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            self.quality
        ]

        _, encoded_img = cv2.imencode(
            '.jpg',
            image,
            encode_param
        )

        # Decode JPEG
        decoded_img = cv2.imdecode(
            encoded_img,
            cv2.IMREAD_COLOR
        )

        # BGR -> RGB
        decoded_img = cv2.cvtColor(
            decoded_img,
            cv2.COLOR_BGR2RGB
        )

        # numpy -> PIL
        decoded_img = Image.fromarray(decoded_img)

        return decoded_img


# MAIN PREPROCESSOR

class WatermarkPreprocessor:
    """
    Kết hợp:
        RGB preprocessing
        FFT preprocessing

    Output:
        rgb_tensor
        fft_tensor
    """

    def __init__(
        self,
        image_size=224,
        train=True,
        use_jpeg_aug=False
    ):

        self.rgb_processor = RGBPreprocessor(
            image_size=image_size,
            train=train
        )

        self.fft_processor = FFTPreprocessor(
            image_size=image_size
        )

        self.use_jpeg_aug = use_jpeg_aug

        if self.use_jpeg_aug:
            self.jpeg_aug = JPEGCompression(
                quality=80
            )

    def __call__(self, image):
        """
        Parameters
        ----------
        image : PIL.Image

        Returns
        -------
        rgb_tensor : torch.Tensor
            Shape: (3,224,224)

        fft_tensor : torch.Tensor
            Shape: (1,224,224)
        """

        
        # JPEG augmentation        
        if self.use_jpeg_aug:
            image = self.jpeg_aug(image)

        
        # RGB branch        
        rgb_tensor = self.rgb_processor(image)

        
        # FFT branch
        fft_tensor = self.fft_processor(image)

        return rgb_tensor, fft_tensor
    


# DATASET CLASS

class WatermarkDataset(Dataset):
    """
    Dataset cho bài toán Watermark Detection.

    Output:
        rgb_tensor
        fft_tensor
        label
    """

    def __init__(
        self,
        root_dir,
        image_size=224,
        train=True,
        use_jpeg_aug=False
    ):
        """
        Parameters
        ----------
        root_dir : str
            Đường dẫn dataset

        image_size : int
            Resize image

        train : bool
            Train / Validation mode

        use_jpeg_aug : bool
            Có dùng JPEG augmentation hay không
        """

        self.root_dir = root_dir

        self.samples = []

        # Class mapping
        #
        # no_watermark -> 0
        # watermark    -> 1
        self.class_to_idx = {
            "no_watermark": 0,
            "watermark": 1
        }

        # Preprocessor
        self.preprocessor = WatermarkPreprocessor(
            image_size=image_size,
            train=train,
            use_jpeg_aug=use_jpeg_aug
        )

        # Load toàn bộ image path
        self.load_dataset()

    def load_dataset(self):
        """
        Load toàn bộ image path và label.
        """

        for class_name, label in self.class_to_idx.items():

            class_dir = os.path.join(
                self.root_dir,
                class_name
            )

            # Kiểm tra folder tồn tại
            if not os.path.exists(class_dir):
                continue

            for image_name in os.listdir(class_dir):

                image_path = os.path.join(
                    class_dir,
                    image_name
                )

                # Lưu sample
                self.samples.append(
                    (image_path, label)
                )

    def __len__(self):
        """
        Số lượng samples.
        """

        return len(self.samples)

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int

        Returns
        -------
        rgb_tensor : Tensor
            Shape: (3,H,W)

        fft_tensor : Tensor
            Shape: (1,H,W)

        label : int
        """

        image_path, label = self.samples[index]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocessing
        rgb_tensor, fft_tensor = self.preprocessor(image)

        return {
            "rgb": rgb_tensor,
            "fft": fft_tensor,
            "label": label
        }

    