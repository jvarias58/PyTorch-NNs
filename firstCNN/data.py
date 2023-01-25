from torch.utils.data import DataLoader


train_data = torchvision.datasets.MNIST(root="./datasets", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root="./datasets", train=False, download=True, transform=torchvision.transforms.ToTensor())


train_d = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_d = DataLoader(dataset=test_data, batch_size=32)