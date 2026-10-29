"""Standalone inference matching InspectAI's 03_patchcore.ipynb.

Uses the final memory bank trained on all 220 normal training images.
This simplified PatchCore implementation emits a raw anomaly score and heatmap;
no binary decision is made without a separately calibrated threshold.
"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor


class PatchCoreInspector:
    def __init__(self, bank_path: str | Path, device: str | None = None):
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.image_size = 256
        self.bank_path = Path(bank_path)
        if not self.bank_path.is_file():
            raise FileNotFoundError(f"Final PatchCore bank missing: {self.bank_path}")

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = create_feature_extractor(
            backbone, return_nodes={"layer2": "layer2", "layer3": "layer3"}
        ).to(self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        # Only load our own trusted training artifact.
        bank = torch.load(self.bank_path, map_location="cpu", weights_only=True)
        if not isinstance(bank, torch.Tensor) or bank.ndim != 2 or bank.shape[1] != 384:
            raise ValueError(f"Unexpected bank format/shape: {getattr(bank, 'shape', None)}")
        if not torch.isfinite(bank).all():
            raise ValueError("The memory bank contains non-finite values")
        self.bank = bank.float().to(self.device)

        self.preprocess = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> tuple[float, np.ndarray]:
        """Return raw image anomaly score, and HxW map at ORIGINAL image size."""
        image = image.convert("RGB")
        original_size = image.size  # width, height
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.encoder(x)
        layer2 = F.avg_pool2d(features["layer2"], kernel_size=3, stride=1, padding=1)
        layer3 = F.avg_pool2d(features["layer3"], kernel_size=3, stride=1, padding=1)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        embeddings = torch.cat([layer2, layer3], dim=1)
        b, channels, h, w = embeddings.shape
        patches = embeddings.permute(0, 2, 3, 1).reshape(-1, channels)

        minimum_distances = []
        for chunk in patches.split(1024):
            minimum_distances.append(torch.cdist(chunk.float(), self.bank).min(dim=1).values)
        raw_map = torch.cat(minimum_distances).reshape(b, h, w)
        score = float(raw_map.flatten(1).max(dim=1).values.item())

        upscaled = F.interpolate(
            raw_map.unsqueeze(1),
            size=(original_size[1], original_size[0]),
            mode="bilinear", align_corners=False,
        )[0, 0].cpu().numpy()
        return score, upscaled
