from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
import os.path as osp
import torchvision.transforms as transforms

class MNIST(Dataset):
    """
    A customized data loader for MNIST.
    """
    def __init__(self,
                 root,
                 transform=None,
                 preload=False):
        """ Intialize the MNIST dataset
        
        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        for i in range(10):
            filenames = glob.glob(osp.join(root, str(i), '*.png'))
            for fn in filenames:
                self.filenames.append((fn, i)) # (filename, label) pair
                
        # if preload dataset into memory
        if preload:
            self._preload()
            
        self.len = len(self.filenames)
                              
    def _preload(self):
        """
        Preload dataset to memory
        """
        self.labels = []
        self.images = []
        for image_fn, label in self.filenames:            
            # load images
            image = Image.open(image_fn)
            # avoid too many opened files bug
            self.images.append(image.copy())
            image.close()
            self.labels.append(label)

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
            label = self.labels[index]
        else:
            # If on-demand data loading
            image_fn, label = self.filenames[index]
            image = Image.open(image_fn)
            
        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            image = self.transform(image)
        # return image and label
        return image, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
    
def MNIST_Dataset_Loader():
    trainset = MNIST(
        root='./dataset/mnist_png/training',
        preload=True, transform=transforms.ToTensor(),
    )
    # Use the torch dataloader to iterate through the dataset
    trainset_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)

    # load the testset
    testset = MNIST(
        root='./dataset/mnist_png/testing',
        preload=True, transform=transforms.ToTensor(),
    )
    # Use the torch dataloader to iterate through the dataset
    testset_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=1)

    print(f"[D] Mnist Train lenght{len(trainset)}")
    print(f"[D] Mnist Test Lenght{len(testset)}")

    return trainset_loader, testset_loader