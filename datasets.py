from torchvision import datasets, transforms
from determined.pytorch import DataLoader
from torch.utils.data import Subset


IMAGE_DATASETS = frozenset(['mnist', 'cifar-10', 'fashion-mnist', 'celeba', 'svhn'])

MNIST_TRANSFORMS = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5), (0.5))
                            ])
        
FashionMNIST_TRANSFORMS = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5), (0.5)),
                            ])

CIFAR10_TRANSFORMS = transforms.Compose([
                            transforms.Resize([32, 32]),
                            transforms.CenterCrop([32, 32]),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])

CELEBA_TRANSFORMS = transforms.Compose([
                            transforms.Resize([64, 64]),
                            transforms.CenterCrop([64, 64]),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])

SVHN_TRANSFORMS = transforms.Compose([
                            transforms.Resize([32, 32]),
                            transforms.CenterCrop([32, 32]),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])

class Datasets:
    """Initialize a training or evaluation dataset
    Args:
        dataset_directory:
            folder to find or save a dataset
        batch_size:
            Batch size as integer
    """
    def __init__(self, dataset_directory: str, batch_size: int) -> None:
        self.batch_size = batch_size
        self.dataset_directory = dataset_directory


    def get_data_loader(self, dataset_string: str, train: bool) -> DataLoader:
        """Load a particular dataset. Possible strings are 'mnist', 'cifar-10', 'fashion-mnist', 'celeba', 'svhn'.
        Args:
            dataset_string:
                String indicating the dataset to load
            train:
                Whether a training or evaluation dataset is requested
        Raises:
            ValueError:
                If dataset_string is unknown
        Returns:
            DataLoader containing the dataset
        """
        if dataset_string not in IMAGE_DATASETS:
            raise ValueError(f'No dataset for string {dataset_string} found. Use one of {IMAGE_DATASETS}.')
        else:
            if dataset_string == 'mnist':
                return DataLoader(
                    dataset=datasets.MNIST(root=self.dataset_directory, train=train, transform=MNIST_TRANSFORMS, download=True),
                    batch_size=self.batch_size,
                    drop_last=True)

            if dataset_string == 'cifar-10':
                return DataLoader(
                    dataset=datasets.CIFAR10(root=self.dataset_directory, train=train, transform=CIFAR10_TRANSFORMS, download=True),
                    batch_size=self.batch_size,
                    drop_last=True)

            if dataset_string == 'fashion-mnist':
                return DataLoader(
                    dataset=datasets.FashionMNIST(root=self.dataset_directory, train=train, transform=FashionMNIST_TRANSFORMS, download=True),
                    batch_size=self.batch_size,
                    drop_last=True)

            if dataset_string == 'celeba':
                data = datasets.ImageFolder(root='/data/ldap/celeba/original_read_only/', transform=CELEBA_TRANSFORMS)
                train_set = Subset(data, range(0, 160000+1))
                test_set = Subset(data, range(160000+1, 180000))

                return DataLoader(
                    train_set if train else test_set,
                    batch_size=self.batch_size,
                    drop_last=True)

            if dataset_string == 'svhn':
                return DataLoader(
                    dataset=datasets.SVHN(root=self.dataset_directory, split='train' if train else 'test', transform=SVHN_TRANSFORMS, download=True),
                    batch_size=self.batch_size,
                    drop_last=True)

