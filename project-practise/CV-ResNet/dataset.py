from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(batch_size=64):

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914,0.4822,0.4465), std=(0.247,0.243,0.261))
        #mean: sequence
        #std: sequence
        #所以Normalize后面的参数要是元组

    ])
    # TODO:
    # 创建 train_dataset / test_dataset (MNIST)
    train_dataset = datasets.CIFAR10(
        root='./data/train',
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = datasets.CIFAR10(
        root='./data/test',
        train=False,
        transform=transform,
        download=True
    )
    # TODO:
    # 用 DataLoader 包装
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
