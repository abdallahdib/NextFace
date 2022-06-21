import face_alignment
import numpy as np
import torch
import cv2


class LandmarksDetectorFAN:
	def __init__(self, mask, device):
		'''
		init landmark detector with given mask on target device
		:param mask: valid mask for the 68 landmarks of shape [n]
		:param device:
		'''
		assert(mask.dim() == 1)
		assert(mask.max().item() <= 67 and mask.min().item() >= 0)

		self.device = device
		self.landmarksDetector = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=self.device)
		self.mask = mask.to(self.device)

	def detect(self, images):
		'''
		detect landmakrs on a batch of images
		:param images: tensor [n, height, width, channels]
		:return: tensor [n, landmarksNumber, 2]
		'''
		#landmarks = torch.zeros([images.shape[0], self.mask.shape[0], 2], device = images.device, dtype = torch.float32)
		assert(images.dim() == 4)
		landmarks = []
		for i in range(len(images)):
			land = self._detect(images[i].detach().cpu().numpy() * 255.0)
			landmarks.append(land)

		torch.set_grad_enabled(True) #it turns out that the landmark detector disables the autograd engine. this line fixes this
		return torch.tensor(landmarks, device = self.device)
	def _detect(self, image):
		arr = self.landmarksDetector.get_landmarks_from_image(image, None)
		if arr is None or len(arr) == 0:
			raise RuntimeError("No landmarks found in image...")
		if len(arr) > 1:
			print('found multiple subjects in image. extracting landmarks for first subject only...')

		landmarks = []
		mask = self.mask.detach().cpu().numpy()
		for preds in arr:

			preds = preds[mask]
			subjectLandmarks = np.array([[p[0], p[1]] for p in preds])
			landmarks.append(subjectLandmarks)
			break #only one subject per frame

		return landmarks[0]
		return torch.tensor(landmarks, device = self.device)

	def drawLandmarks(self, image, landmarks):
		'''
		draw landmakrs on top of image (for debug)
		:param image: tensor representing the image [h, w, channels]
		:param landmarks:  tensor representing the image landmarks [n, 2]
		:return:
		'''
		assert(image.dim() == 3 and landmarks.dim() == 2 and landmarks.shape[-1] ==2)
		clone = np.copy(image.detach().cpu().numpy() * 255.0)
		land = landmarks.cpu().numpy()
		for x in land:
			cv2.circle(clone, (int(x[0]), int(x[1])), 1, (0, 0, 255), -1)
		return clone
