import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms


def load_image(file_path: str) -> Tensor:
    """Load an image from a file and convert it to a PyTorch tensor.

    The image is loaded in RGB format, converted to a tensor with shape (C, H, W),
    and normalized to the range [0, 1].

    Args:
        file_path (str): Absolute path to the image file.

    Returns:
        Tensor: A PyTorch tensor of shape (C, H, W) with pixel values in [0, 1].
    """
    # Load image using PIL
    img = Image.open(file_path)

    # Convert to RGB (handles grayscale or RGBA images)
    img = img.convert("RGB")

    # Define transformation pipeline
    transform = transforms.Compose(
        [
            transforms.ToTensor()  # Converts PIL Image (H, W, C) to tensor (C, H, W),
            # normalizes pixels from [0, 255] to [0, 1]
        ]
    )

    # Apply transformation
    img_tensor = transform(img)

    return img_tensor


if __name__ == "__main__":
    file_path = "img.jpg"
    img = load_image(file_path)
    print(img.shape)  # Example output: torch.Size([3, 334, 500])
    print(type(img))  # Output: <class 'torch.Tensor'>
    print(img.min(), img.max())  # Verify pixel range: [0, 1]
