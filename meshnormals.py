import torch

class MeshNormals:

    def __init__(self, device, faces, vertexIndex, vertexFaceNeighbors):
        assert(vertexIndex is not None)
        assert(vertexFaceNeighbors is not None)

        self.device = device
        self.faces = faces
        self.vertexIndex = []
        self.vertexFaceNeighbors = []
        if vertexIndex is not None and vertexFaceNeighbors is not None:
            for i in range(len(vertexIndex)):
                self.vertexIndex.append(torch.tensor(vertexIndex[i]).to(self.device))
                self.vertexFaceNeighbors.append(torch.tensor(vertexFaceNeighbors[i]).to(self.device))

    def computeNormals(self, vertices):
        '''
        compute vertices normal
        :param vertices: [..., verticesNumber, 3]
        :return: normalized normal vectors [..., verticesNumber, 3]
        '''

        faces = self.faces
        assert(faces is not None)
        assert(vertices.shape[-1] == 3)

        v1 = vertices[..., faces[:, 0], :]
        v2 = vertices[..., faces[:, 1], :] - v1
        v3 = vertices[..., faces[:, 2], :] - v1
        faceNormals = torch.cross(v2, v3, dim=vertices.dim() - 1)

        normals = torch.zeros_like(vertices)
        for (ni, vi) in zip(self.vertexFaceNeighbors, self.vertexIndex):
            vc4 = faceNormals[..., ni, :]
            vc4 = torch.mean(vc4, -2)
            normals[..., vi, :] = vc4

        return torch.nn.functional.normalize(normals, 2, -1)

