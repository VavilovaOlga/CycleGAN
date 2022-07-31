from torch.utils.data import Dataset
import torchvision.transforms as tt
from skimage.io import imread
from skimage.color import gray2rgb


class ImageDataset(Dataset):

    def __init__(self, files, stats=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 image_size=256, resize_size=256, padding=0):
        super().__init__()
        self.files = files
        self.stats = stats
        self.image_size = image_size
        self.resize_size = resize_size
        self.padding = padding
        self.len_ = len(self.files)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = imread(file)
        return image

    def __getitem__(self, index):
        transforms = tt.Compose([
            tt.ToPILImage(),
            tt.Resize(self.resize_size),
            tt.Pad(self.padding),
            tt.CenterCrop(self.image_size),
            tt.RandomHorizontalFlip(),
            tt.ToTensor(),
            tt.Normalize(*self.stats)])

        x = self.load_sample(self.files[index])
        if x.ndim == 2:
            x = gray2rgb(x)
        x = transforms(x)
        return x
