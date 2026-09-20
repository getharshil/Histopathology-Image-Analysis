"""Image preprocessing transforms for histopathology images."""
from torchvision import transforms

# ImageNet normalization (used with pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224, augmentation: str = "moderate") -> transforms.Compose:
    """Get training transforms with histopathology-appropriate augmentation.
    
    Histopathology rationale for each transform:
    - HorizontalFlip/VerticalFlip: Tissue orientation is arbitrary under microscope.
    - Rotation: Slide can be placed at any angle.
    - ColorJitter: Accounts for stain variation between labs/scanners.
    - Normalize: ImageNet stats for transfer learning compatibility.
    
    Transforms that are NOT used and why:
    - RandomErasing: Could remove critical tissue features.
    - Perspective: Not biologically meaningful for flat tissue sections.
    """
    base = [transforms.Resize((image_size, image_size))]
    
    if augmentation == "none":
        aug = []
    elif augmentation == "moderate":
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ]
    elif augmentation == "strong":
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.04),
        ]
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation}")
    
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    
    return transforms.Compose(base + aug + normalize)


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Get evaluation transforms (no augmentation — only resize + normalize)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(tensor):
    """Reverse ImageNet normalization for visualization."""
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)
