import random
from torchvision.transforms import functional as F

class Compose(object):
    """组合transform"""
    def __init__(self,transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    """PIL to Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
    
class RandomHorizontalFlip(object):
    """Random horizontal flip"""
    def __init__(self, prob=0.5):  
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            _, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            # bbox: xmin ymin xmax ymax
            bbox[:,[0,2]] = width - bbox[:,[2,0]] # flip the x coordinate
            target["boxes"] = bbox
        return image, target