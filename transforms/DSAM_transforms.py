from torchvision import transforms

def get_transforms(output_size):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(output_size),
        transforms.ColorJitter(.4, .4, .4, .4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((output_size, output_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    data_transforms = {
        'train': train_transform,
        'val': eval_transform,
        'test': eval_transform
    }

    return data_transforms
