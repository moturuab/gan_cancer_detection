import torch
import torchvision.datasets as dsets
from torchvision import transforms
from MRIDataset import MRIDataset

class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.im_height = image_size['height']
        self.im_width = image_size['width']
        self.batch = batch_size
        self.shuf = shuf
        self.train = train

    def transform(self, totensor, normalize):
        options = []
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_wbmri(self):
        transforms = self.transform(True, False)
        dataset = MRIDataset(csv_file='./annotations.csv',root_dir=self.path)
        return dataset

    def loader(self):
        if self.dataset == 'wbmri':
            dataset = self.load_wbmri()

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=self.shuf,
                                             num_workers=2,
                                             drop_last=True)
        return loader
