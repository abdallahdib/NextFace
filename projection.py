import numpy as np
import torch
import math
import cv2

def isRotationMatrix(R):
    """
    return true if the R is a rotation matrix else False (M . T^T = I and det(M) = 1)
    """
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return False
    isIdentity = np.allclose(R.dot(R.T), np.identity(R.shape[0], float))
    isDetEqualToOne = np.allclose(np.linalg.det(R), 1)
    return isIdentity and isDetEqualToOne


def eulerToRodrigues(angles):
    """
    convert euler angles to rodrigues
    """
    rotx = np.array([[1, 0, 0],
                     [0, math.cos(angles[0]), -math.sin(angles[0])],
                     [0, math.sin(angles[0]), math.cos(angles[0])]
                     ])

    roty = np.array([[math.cos(angles[1]), 0, math.sin(angles[1])],
                     [0, 1, 0],
                     [-math.sin(angles[1]), 0, math.cos(angles[1])]
                     ])

    rotz = np.array([[math.cos(angles[2]), -math.sin(angles[2]), 0],
                     [math.sin(angles[2]), math.cos(angles[2]), 0],
                     [0, 0, 1]
                     ])

    R = np.dot(rotz, np.dot(roty, rotx))
    rotVec, _ = cv2.Rodrigues(R)
    return rotVec


def rodrigues2Euler(rotation_vector):
    """
    retrieve euler angles from rodrigues matrix
    """
    rMat, _ = cv2.Rodrigues(rotation_vector)
    assert (rMat.shape[0] == 3 and rMat.shape[1] == 3 and isRotationMatrix(rMat))
    roll = math.atan2(rMat[2, 1], rMat[2, 2])
    pitch = math.atan2(-rMat[2, 0], math.sqrt(rMat[0, 0] * rMat[0, 0] + rMat[1, 0] * rMat[1, 0]))
    yaw = math.atan2(rMat[1, 0], rMat[0, 0])
    return np.array([roll, pitch, yaw])


def estimateCameraPosition(focalLength, image_center, landmarks, vertices, rotAngles, translation):
    '''
    estimate the camera position (rotation and translation) using perspective n points pnp
    :param focalLength: tensor representing the camera focal length of shape [n]
    :param image_center: tensor representing the camera center point [n, 2]
    :param landmarks: tensor representing the 2d landmarks in pixel coordinates system [n, verticesNumber, 2]
    :param vertices: tensor representing the 3d coordinate position of the landmarks  [n, verticesNumber, 3]
    :param rotAngles: the initial rotation angles [n, 3]
    :param translation: the initial translation angles [n, 3]
    :return: estimated rotation [n, 3] , estimated translations  [n, 3]
    '''
    assert (focalLength.dim() == 1 and
            image_center.dim() == 2 and
            image_center.shape[-1] == 2 and
            landmarks.dim() == 3 and landmarks.shape[-1] == 2 and
            vertices.dim() == 3 and vertices.shape[-1] == 3 and
            rotAngles.dim() == 2 and rotAngles.shape[-1] == 3 and
            translation.dim() == 2 and translation.shape[-1] == 3)
    assert (focalLength.shape[0] == image_center.shape[0] == landmarks.shape[0] == vertices.shape[0] == rotAngles.shape[0] == translation.shape[0])
    rots = []
    transs = []
    for i in range(focalLength.shape[0]):
        rot, trans = solvePnP(focalLength[i].item(),
                              image_center[i].detach().cpu().numpy(),
                              vertices[i],
                              landmarks[i],
                              rotAngles[i],
                              translation[i])
        rots.append(rot)
        transs.append(trans)
    return torch.tensor(rots, device=vertices.device, dtype=torch.float32), torch.tensor(transs, device=vertices.device,
                                                                                         dtype=torch.float32)


def solvePnP(focalLength, imageCenter, vertices, pixels, rotAngles, translation):
    """
    Finds an object pose from 3D vertices  <-> 2D pixels correspondences
    Inputs:
     * focalLength: camera focal length
     * imageCenter: center [x, y] of the image
     * vertices: float tensor [n, 3], of vertices
     * pixels: float tensor [n, 2] of corresponding pixels
     * rotAngles: initial euler angles
     * translation: initial translation vector
    """

    cameraMatrix = np.array(
        [[focalLength, 0, imageCenter[0]],
         [0, focalLength, imageCenter[1]],
         [0, 0, 1]], dtype="double"
    )

    success, rotVec, transVec = cv2.solvePnP(vertices.clone().detach().cpu().numpy(),
                                             pixels[:, None].detach().cpu().numpy(),
                                             cameraMatrix,
                                             np.zeros((4, 1)),
                                             eulerToRodrigues(rotAngles.detach().cpu().numpy()),
                                             translation.detach().cpu().numpy(),
                                             True,
                                             flags=cv2.SOLVEPNP_ITERATIVE)
    assert success, "failed to estimate the pose using pNp"

    rotAngles = rodrigues2Euler(rotVec)

    if rotAngles[0] < 0.:
        rotAngles[0] += 2. * math.pi

    translation = transVec.reshape((3,))
    return rotAngles, translation
