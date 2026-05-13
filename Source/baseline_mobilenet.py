"""
Baseline model 2: MobileNetV3 cho phát hiện watermark.
Mô hình CNN nhẹ hơn ResNet18, phù hợp cho edge devices.
"""
import torch
import torch.nn as nn
from torchvision import models


class BaselineMobileNet(nn.Module):
    """
    Baseline MobileNetV3 Small cho phân loại watermark.
    Nhẹ hơn ResNet18, feature_dim=576, nhanh hơn nhưng có thể kém chính xác hơn.
    """
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(BaselineMobileNet, self).__init__()

        # Load pretrained MobileNetV3 Small từ ImageNet
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.feature_dim = 576

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Classifier: Flatten -> Dropout -> FC -> ReLU -> Dropout -> FC
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """Forward pass: feature extraction + classification."""
        features = self.features(x)
        features = self.avgpool(features)
        logits = self.classifier(features)
        return logits


def create_baseline_mobilenet(num_classes=2, pretrained=True, dropout=0.5):
    """Factory function để tạo BaselineMobileNet model."""
    return BaselineMobileNet(num_classes=num_classes, pretrained=pretrained, dropout=dropout)


class WatermarkDetectorMobileNet:
    """
    Wrapper class để dễ dàng load model và thực hiện inference.
    Cung cấp method detect() cho việc phát hiện watermark trên ảnh đơn lẻ.
    """
    def __init__(self, model_path=None, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = create_baseline_mobilenet(num_classes=2, pretrained=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        if model_path and __import__('os').path.exists(model_path):
            self.load_weights(model_path)

    def load_weights(self, path):
        """Load weights từ checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded weights from {path}")

    def detect(self, image_path, return_prob=False):
        """
        Phát hiện watermark trong ảnh.
        Args:
            image_path: Đường dẫn đến ảnh
            return_prob: Có trả về xác suất của cả 2 lớp không
        Returns:
            (label, confidence) hoặc (label, confidence, probabilities)
        """
        import cv2
        from torchvision import transforms

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transform ảnh để phù hợp với model
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()

        label = "Watermark" if pred == 1 else "No Watermark"

        if return_prob:
            return label, confidence, probs[0].cpu().numpy()
        return label, confidence