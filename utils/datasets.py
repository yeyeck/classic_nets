import torch
from torch.utils.data import DataLoader, Dataset, dataset
from PIL import Image
from torchvision import transforms
import os, glob


class Caltech256(Dataset):
    def __init__(self, img_dir, classes, transform = None):

        img_pths = sorted(glob.glob(img_dir  + os.sep + '**' + os.sep + '**.jpg')) 
        assert img_pths, 'no jpg file in ' + img_dir
        self.img_pths = img_pths
        self.classes = classes
        self.transform = transform 
    
    def __len__(self):
        return len(self.img_pths)
    
    def __getitem__(self, idx):
        img_pth = self.img_pths[idx]
        cls_name = img_pth.split(os.sep)[-2]
        
        label = self.classes.index(cls_name)
        image = Image.open(img_pth).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloader(img_dir, train=False, batch_size=128, num_workers=0):
    train_dir = os.path.join(img_dir, 'train')
    classes = sorted(os.listdir(train_dir))
    if train:
        transform = transforms.Compose([
            transforms.Resize(size=300),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomCrop(size=224)]), p=0.5),
            transforms.Resize(size=(224, 224)),
            transforms.RandomErasing(p=0.4)    
        ])
        img_dir = train_dir
        

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)), 
        ])
        img_dir = os.path.join(img_dir, 'val')

    dataset = Caltech256(img_dir, classes, transform)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)



if __name__ == '__main__':
    import random
    classes = sorted(os.listdir('/caltech20/train'))
    print(classes)
    dataset = Caltech256('/caltech20/train', classes=classes)
    n = dataset.__len__()
    for i in range(100):
        img_pth, _, label = dataset.__getitem__(random.randint(0, n))
        print(img_pth, label)
