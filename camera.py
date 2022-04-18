import torch
import numpy as np



class Camera:

    def __init__(self, device):
        self.device = device

        self.rotXm1 = torch.tensor(np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), dtype=torch.float, device=device)
        self.rotXm2 = torch.tensor(np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]]), dtype=torch.float, device=device)
        self.rotXm3 = torch.tensor(np.array([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]), dtype=torch.float, device=device)

        self.rotYm1 = torch.tensor(np.array([[1., 0., 0.], [0., 0., 0.], [0., 0., 1.]]), dtype=torch.float, device=device)
        self.rotYm2 = torch.tensor(np.array([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]]), dtype=torch.float, device=device)
        self.rotYm3 = torch.tensor(np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]), dtype=torch.float, device=device)

        self.rotZm1 = torch.tensor(np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.]]), dtype=torch.float, device=device)
        self.rotZm2 = torch.tensor(np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]]), dtype=torch.float, device=device)
        self.rotZm3 = torch.tensor(np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]), dtype=torch.float, device=device)

    def computeTransformation(self, rotation, translation):
        '''
        create a transformation matrix from rotation and translation
        rotation: [n, 3]
        translation: [n, 3]
        return: transformation matrix [n, 4, 3]
        '''

        assert (rotation.dim() == 2 and rotation.shape[-1] == 3)
        assert(translation.dim() == 2 and translation.shape[-1] == 3)

        rotx = torch.cos(rotation[..., :1, None]).expand(-1, 3, 3) * self.rotXm1 \
               + torch.sin(rotation[..., :1, None]).expand(-1, 3, 3) * self.rotXm2 \
               + self.rotXm3
        roty = torch.cos(rotation[..., 1:2, None]).expand(-1, 3, 3) * self.rotYm1 \
               + torch.sin(rotation[..., 1:2, None]).expand( -1, 3, 3) * self.rotYm2 \
               + self.rotYm3
        rotz = torch.cos(rotation[..., 2:, None]).expand(-1, 3, 3) * self.rotZm1 \
               + torch.sin(rotation[..., 2:, None]).expand(-1, 3, 3) * self.rotZm2 \
               + self.rotZm3

        rotMatrix = torch.matmul(rotz, torch.matmul(roty, rotx))
        transformation = torch.cat((rotMatrix, translation[ :, :, None]), -1)
        return transformation

    def transformVertices(self, vertices, translation, rotation):
        '''
        transform vertices by the rotation and translation vector
        :param vertices: tensor [n, verticesNumber, 3]
        :param translation:  tensor [n, 3]
        :param rotation: tensor [n, 3]
        :return: transformed vertices [n, verticesNumber, 3]
        '''
        assert (vertices.dim() == 3 and vertices.shape[-1] == 3)

        transformationMatrix = self.computeTransformation(rotation, translation)
        ones = torch.ones([vertices.shape[0], vertices.shape[1], 1], dtype = torch.float, device = vertices.device)
        vertices = torch.cat((vertices, ones), -1)
        framesNumber = transformationMatrix.shape[0]
        verticesNumber = vertices.shape[1]
        out = torch.matmul(transformationMatrix.view(1, framesNumber, 1, 3, 4),
                           vertices.view(framesNumber, verticesNumber, 4, 1)).view(1, framesNumber, verticesNumber, 3)
        return out[0]