import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Set image size
IMAGE_SIZE = 16

# Compose the transforms
composed = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# Load the datasets
dataset_train = dsets.FashionMNIST(root='./fashion/data', train=True, transform=composed, download=True)
dataset_val = dsets.FashionMNIST(root='./fashion/data', train=False, transform=composed, download=True)

# Show 3 validation samples in one row
fig, axes = plt.subplots(1, 3, figsize=(9, 3))  # 1 row, 3 columns

for i in range(3):
    image_tensor, label = dataset_val[i]
    image = image_tensor.squeeze()  # remove channel dimension
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()