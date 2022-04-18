import sys
from os import walk
import numpy as np
import torch
import cv2
import os


def saveImage(image, fileName, gamma = 2.2):
    '''
    save image to drive
    :param image: float tensor [w, h, 3/4]
    :param fileName: path to where to save the image
    :param gamma: gamma correction
    :return:
    '''

    import pyredner
    pyredner.imwrite(image.cpu().detach(), fileName, gamma = gamma)

def overlayImage(background, image):
    '''
    overlay image on top of background image an image on a background image.
    :param background: float tensor [width,height,3]
    :param image: float tensor [width, height, 4]
    :return: float tensor [width, height, 3]
    '''
    assert(torch.is_tensor(background) and torch.is_tensor(image) and background.dim() == 3 and image.dim() == 3 and background.shape[0] == image.shape[0] and background.shape[1] == image.shape[1])
    assert(background.shape[-1] == 3 and image.shape[-1] == 4)
    from torchvision import transforms
    background = transforms.ToPILImage()(background.permute(2, 1, 0).clone().detach().cpu()).convert("RGB")
    image = transforms.ToPILImage()(torch.clamp(image.permute(2, 1, 0), 0, 1).clone().detach().cpu()).convert("RGBA")
    background.paste(image, (0, 0), image)
    return transforms.ToTensor()(background).permute(2, 1, 0)

def resizeImage(image, targetResolution):
    '''
    resize an image (as numpy array) to the target resolution
    :param image: numpy array [h, w, 4/3/1]
    :param targetResolution: int > 0
    :return: numpy array [h, w, 4/3/1]
    '''
    assert(image is not None and isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[-1] == 3 or image.shape[-1] == 4 or image.shape[-1] == 1)
    dmax = max(image.shape[0], image.shape[1])

    if (dmax > targetResolution):
        print("[INFO] resizing input image to fit:", targetResolution,"px resolution...")
        if (image.shape[0] > image.shape[1]):
            scale = float(targetResolution) / float(image.shape[0])
        else:
            scale = float(targetResolution) / float(image.shape[1])
        img = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_CUBIC )
    else:
        return image
    return img

class Image:

    def __init__(self, path, device, maxRes = 512):
        '''
        class that represent a single image  as a pytorch tensor [1, h, w, channels]
        :param path: the path to the image
        :param device: where to store the image ('cpu' or 'cuda')
        :param maxRes: maximum allowed resolution (depending on the gpu/cpu memory and speed, this limit can be increased or removed)
        '''
        assert(maxRes > 0)
        print('loading image from path: ', path)
        self.device = device
        numpyImage = cv2.imread(path)[..., 0:3]
        assert (numpyImage is not None)
        numpyImage = resizeImage(cv2.cvtColor(numpyImage, cv2.COLOR_BGR2RGB), int(maxRes))
        self.tensor = (torch.from_numpy(numpyImage).to(self.device).to(dtype=torch.float32) / 255.0).unsqueeze(0)
        self.height = numpyImage.shape[0]
        self.width = numpyImage.shape[1]
        self.channels = numpyImage.shape[2]
        self.gamma = 2.2
        self.center = torch.tensor([ self.width / 2, self.height / 2], dtype = torch.float32, device = self.device).reshape(1, -1)
        self.imageName = os.path.basename(path)

class ImageFolder:

    def __init__(self, path, device, maxRes = 512):
        '''
        class that represent images in a given path
        :param path: the path to the image
        :param device: where to store the image ('cpu' or 'cuda')
        '''
        print('loading images from path: ', path)
        self.device = device
        self.tensor = None
        self.imageNames = []
        supportedFormats = ['.jpg', '.jpeg', '.png']

        filenames = next(walk(path), (None, None, []))[2]
        width = None
        height = None
        ct = 0

        assert (len(filenames) > 0)  # no images found in the given directory
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in supportedFormats:
                image = Image(path + '/' + filename, device, maxRes)

                if width is None:
                    width = image.width
                    height = image.height
                    self.tensor = torch.zeros([len(filenames), height, width, image.channels], device = self.device)
                    self.center = torch.zeros([len(filenames), 2], device = self.device)

                assert image.width == width and image.height == height

                self.width = image.width
                self.height = image.height
                self.channels = image.channels
                self.tensor[ct] = image.tensor[0].clone().detach()
                self.center[ct] = image.center[0].clone().detach()
                self.imageNames.append(image.imageName)
                image = None

                ct += 1


        import gc
        gc.collect()
        self.gamma = 2.2

    @property
    def asNumpyArray(self):
        return self.tensor.detach().cpu().numpy() * 255.0

if __name__ == "__main__":
    pass