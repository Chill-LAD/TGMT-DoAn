import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from config import Config, SEAttentionConfig, ModelConfig


class SEAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FrequencyBranch(nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        super(FrequencyBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RGBBackbone(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True):
        super(RGBBackbone, self).__init__()

        if backbone_name == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone_name == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone_name == "mobilenet_v3_small":
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            feature_dim = 576
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.feature_dim = feature_dim
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return x


class HybridWatermarkModel(nn.Module):
    def __init__(self, num_classes=2, backbone="resnet18", pretrained=True,
                 dropout=0.5, use_se_attention=True):
        super(HybridWatermarkModel, self).__init__()

        self.rgb_backbone = RGBBackbone(backbone, pretrained)
        self.freq_branch = FrequencyBranch(out_channels=128)

        rgb_dim = self.rgb_backbone.feature_dim
        freq_dim = 128
        fused_dim = rgb_dim + freq_dim

        self.fusion_fc = nn.Linear(fused_dim, 512)

        self.use_se_attention = use_se_attention
        if use_se_attention:
            self.se_attention = SEAttention(512, reduction=SEAttentionConfig.reduction)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb_input, freq_input=None):
        rgb_features = self.rgb_backbone(rgb_input)

        if freq_input is not None:
            freq_features = self.freq_branch(freq_input)
            fused_features = torch.cat([rgb_features, freq_features], dim=1)
        else:
            freq_features = torch.zeros(rgb_features.size(0), 128).to(rgb_features.device)
            fused_features = torch.cat([rgb_features, freq_features], dim=1)

        fused_features = self.fusion_fc(fused_features)

        if self.use_se_attention:
            fused_features_2d = fused_features.unsqueeze(-1).unsqueeze(-1)
            fused_features_2d = self.se_attention(fused_features_2d)
            fused_features = fused_features_2d.squeeze(-1).squeeze(-1)

        logits = self.classifier(fused_features)

        return logits

    def freeze_backbone(self):
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.rgb_backbone.parameters():
            param.requires_grad = True


def create_model(num_classes=2, backbone="resnet18", pretrained=True,
                dropout=0.5, use_se_attention=True):
    model = HybridWatermarkModel(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
        use_se_attention=use_se_attention
    )
    return model


class WatermarkDetector:
    def __init__(self, model_path=None, device=None):
        self.device = device or Config.device
        self.model = create_model(num_classes=2, backbone=ModelConfig.backbone,
                                  pretrained=ModelConfig.pretrained,
                                  dropout=ModelConfig.dropout)
        self.model.to(self.device)
        self.model.eval()

        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)

    def load_weights(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded weights from {path}")

    def detect(self, image_path, return_prob=False):
        import cv2
        from dataset import compute_fft_spectrum

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = Config.image_size, Config.image_size
        image_resized = cv2.resize(image, (w, h))

        rgb_tensor = torch.from_numpy(image_resized).float() / 255.0
        rgb_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(rgb_tensor.permute(2, 0, 1))
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)

        freq_tensor = compute_fft_spectrum(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(rgb_tensor, freq_tensor)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred].item()

        label = "Watermark" if pred == 1 else "No Watermark"

        if return_prob:
            return label, confidence, probs[0].cpu().numpy()
        return label, confidence


from torchvision import transforms
import os