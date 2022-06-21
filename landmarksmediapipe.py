import mediapipe as mp
import numpy as np
import torch
import cv2

class LandmarksDetectorMediapipe:
	def __init__(self, mask, device, is_video=False, refine_landmarks=False):
		'''
		init landmark detector with given mask on target device
		:param mask: valid mask for the 468 landmarks of shape [n]
		:param device:
		:param is_video: set to true if passing frames sequentially in order
		:param refine_landmarks: if the facemesh attention module should be applied. Note: requires mediapipe 0.10
		'''
		assert(mask.dim() == 1)
		assert(mask.max().item() <= 467 and mask.min().item() >= 0)

		self.device = device
		mp_face_mesh = mp.solutions.face_mesh

		if refine_landmarks:
			try:
				self.landmarksDetector = mp_face_mesh.FaceMesh(
					static_image_mode=not is_video,
					refine_landmarks=True,
					min_detection_confidence=0.5,
					min_tracking_confidence=0.5,
				)
			except KeyError:
				raise KeyError('Refine landmarks is only available with the latest version of mediapipe')

		else:
			self.landmarksDetector = mp_face_mesh.FaceMesh(
				static_image_mode=not is_video,
				min_detection_confidence=0.5,
				min_tracking_confidence=0.5,
			)

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
			land = self._detect((images[i].detach().cpu().numpy() * 255.0).astype('uint8'))
			landmarks.append(land)

		torch.set_grad_enabled(True) #it turns out that the landmark detector disables the autograd engine. this line fixes this
		return torch.tensor(landmarks, device = self.device)

	def _detect(self, image):

		height, width, _ = image.shape

		results = self.landmarksDetector.process(image)
		mask = self.mask.detach().cpu().numpy()
		multi_face_landmarks = results.multi_face_landmarks

		if multi_face_landmarks:
			face_landmarks = multi_face_landmarks[0]
			landmarks = np.array(
				[(lm.x * width, lm.y * height) for lm in face_landmarks.landmark]
			)
		else:
			raise RuntimeError('No face was found in this image')

		return landmarks[mask]

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

