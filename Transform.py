from torchvision import transforms


def transform_for_training(image_shape):
    return transforms.Compose(
       [transforms.ToPILImage(),
        transforms.Resize(image_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )


def transform_for_infer(image_shape):
    return transforms.Compose(
       [transforms.ToPILImage(),
        transforms.Resize(image_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )