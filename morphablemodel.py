from utils import loadDictionaryFromPickle, writeDictionaryToPickle
from normalsampler import NormalSampler
from meshnormals import MeshNormals
import numpy as np
import torch
import h5py
import sys
import os

class MorphableModel:

    def __init__(self, path, textureResolution = 256, trimPca = False, landmarksPathName = 'landmark_62_mp.txt', device='cuda'):
        '''
        a statistical morphable model is a generative model that can generate faces with different identity, expression and skin reflectance
        it is mainly composed of an orthogonal basis (eigen vectors) obtained from applying principal component analysis (PCA) on a set of face scans.
        a linear combination of these eigen vectors produces different type shape and skin
        :param path: drive path of where the data of the morphable model is saved
        :param textureResolution: the resolution of the texture used for diffuse and specular reflectance
        :param trimPca: if True keep only a subset of the PCA basis
        :param landmarksPathName: a text file conains the association between the 2d pixel position and the 3D points in the mesh
        :param device: where to store the morphableModel data (cpu or gpu)
        '''
        assert textureResolution == 256 or textureResolution == 512 or textureResolution == 1024 or textureResolution == 2048 #can handle only 256 or 512 texture res
        self.shapeBasisSize = 199
        self.albedoBasisSize = 145
        self.expBasisSize = 100
        self.device = device
        pathH5Model = path + '/model2017-1_face12_nomouth.h5'
        pathAlbedoModel = path + '/albedoModel2020_face12_albedoPart.h5'
        pathUV = path + '/uvParametrization.' + str(textureResolution) + '.pickle'
        pathLandmarks = path + '/' + landmarksPathName

        pathPickleFileName = path + '/morphableModel-2017.pickle'
        pathNormals = path + '/normals.pickle'

        if os.path.exists(pathPickleFileName) == False:
            print("Loading Basel Face Model 2017 from " + pathH5Model + "... this may take a while the first time... The next runtime it will be faster...")

            if os.path.exists(pathH5Model) == False:
                print('[Error] to use the library, you have to install basel morphable  face model 2017 from: https://faces.dmi.unibas.ch/bfm/bfm2017.html', file=sys.stderr, flush=True)
                print('Fill the form on the link and you will get instant download link into your inbox.', file=sys.stderr, flush=True)
                print('Download  "model2017-1_face12_nomouth.h5" and put it inside ',path, ' and run again...', file=sys.stderr, flush=True)
                exit(0)

            self.file = h5py.File(pathH5Model, 'r')
            assert(self.file is not None)

            print("loading shape basis...")
            self.shapeMean = torch.Tensor(self.file["shape"]["model"]["mean"]).reshape(-1, 3).to(device).float()
            self.shapePca = torch.Tensor(self.file["shape"]["model"]["pcaBasis"]).reshape(-1, 3, self.shapeBasisSize).to(device).float().permute(2, 0, 1)
            self.shapePcaVar = torch.Tensor(self.file["shape"]["model"]["pcaVariance"]).reshape(self.shapeBasisSize).to(device).float()

            print("loading expression basis...")
            self.expressionPca = torch.Tensor(self.file["expression"]["model"]["pcaBasis"]).reshape(-1, 3, self.expBasisSize).to(device).float().permute(2, 0, 1)
            self.expressionPcaVar = torch.Tensor(self.file["expression"]["model"]["pcaVariance"]).reshape(self.expBasisSize).to(device).float()
            self.faces = torch.Tensor(np.transpose(self.file["shape"]["representer"]["cells"])).reshape(-1, 3).to(device).long()
            self.file.close()

            print("Loading Albedo model from " + pathAlbedoModel + "...")
            if os.path.exists(pathAlbedoModel) == False:
                print('[ERROR] Please install the albedo model from the link below, put it inside', path, 'and run again: https://github.com/waps101/AlbedoMM/releases/download/v1.0/albedoModel2020_face12_albedoPart.h5', file=sys.stderr, flush=True)
                exit(0)

            self.file = h5py.File(pathAlbedoModel, 'r')
            assert(self.file is not None)

            self.diffuseAlbedoMean = torch.Tensor(self.file["diffuseAlbedo"]["model"]["mean"]).reshape(-1, 3).to(device).float()
            self.diffuseAlbedoPca = torch.Tensor(self.file["diffuseAlbedo"]["model"]["pcaBasis"]).reshape(-1, 3, self.albedoBasisSize).to(device).float().permute(2, 0, 1)
            self.diffuseAlbedoPcaVar = torch.Tensor(self.file["diffuseAlbedo"]["model"]["pcaVariance"]).reshape(self.albedoBasisSize).to(device).float()

            self.specularAlbedoMean = torch.Tensor(self.file["specularAlbedo"]["model"]["mean"]).reshape(-1, 3).to(device).float()
            self.specularAlbedoPca = torch.Tensor(self.file["specularAlbedo"]["model"]["pcaBasis"]).reshape(-1, 3, self.albedoBasisSize).to(device).float().permute(2, 0, 1)
            self.specularAlbedoPcaVar = torch.Tensor(self.file["specularAlbedo"]["model"]["pcaVariance"]).reshape(self.albedoBasisSize).to(device).float()
            self.file.close()

            #save to pickle for future loading
            dict = {'shapeMean': self.shapeMean.cpu().numpy(),
                    'shapePca': self.shapePca.cpu().numpy(),
                    'shapePcaVar': self.shapePcaVar.cpu().numpy(),

                    'diffuseAlbedoMean': self.diffuseAlbedoMean.cpu().numpy(),
                    'diffuseAlbedoPca': self.diffuseAlbedoPca.cpu().numpy(),
                    'diffuseAlbedoPcaVar': self.diffuseAlbedoPcaVar.cpu().numpy(),

                    'specularAlbedoMean': self.specularAlbedoMean.cpu().numpy(),
                    'specularAlbedoPca': self.specularAlbedoPca.cpu().numpy(),
                    'specularAlbedoPcaVar': self.specularAlbedoPcaVar.cpu().numpy(),

                    'expressionPca': self.expressionPca.cpu().numpy(),
                    'expressionPcaVar': self.expressionPcaVar.cpu().numpy(),
                    'faces': self.faces.cpu().numpy()}
            writeDictionaryToPickle(dict, pathPickleFileName)
        else:
            print("Loading Basel Face Model 2017 from " + pathPickleFileName + "...")

            dict = loadDictionaryFromPickle(pathPickleFileName)
            self.shapeMean = torch.tensor(dict['shapeMean']).to(device)
            self.shapePca = torch.tensor(dict['shapePca']).to(device)
            self.shapePcaVar = torch.tensor(dict['shapePcaVar']).to(device)

            self.diffuseAlbedoMean = torch.tensor(dict['diffuseAlbedoMean']).to(device)
            self.diffuseAlbedoPca = torch.tensor(dict['diffuseAlbedoPca']).to(device)
            self.diffuseAlbedoPcaVar = torch.tensor(dict['diffuseAlbedoPcaVar']).to(device)
            
            self.specularAlbedoMean = torch.tensor(dict['specularAlbedoMean']).to(device)
            self.specularAlbedoPca = torch.tensor(dict['specularAlbedoPca']).to(device)
            self.specularAlbedoPcaVar = torch.tensor(dict['specularAlbedoPcaVar']).to(device)

            self.expressionPca = torch.tensor(dict['expressionPca']).to(device)
            self.expressionPcaVar = torch.tensor(dict['expressionPcaVar']).to(device)
            self.faces = torch.tensor(dict['faces']).to(device)


        if trimPca:
            newDim = min(80,
                         self.shapePca.shape[0],
                         self.diffuseAlbedoPca.shape[0],
                         self.specularAlbedoPcaVar.shape[0],
                         self.expressionPca.shape[0])

            self.shapePca = self.shapePca[0:newDim, ...]
            self.shapePcaVar = self.shapePcaVar[0:newDim, ...]

            self.diffuseAlbedoPca = self.diffuseAlbedoPca[0:newDim, ...]
            self.diffuseAlbedoPcaVar = self.diffuseAlbedoPcaVar[0:newDim, ...]

            self.specularAlbedoPca = self.specularAlbedoPca[0:newDim, ...]
            self.specularAlbedoPcaVar = self.specularAlbedoPcaVar[0:newDim, ...]

            self.expressionPca = self.expressionPca[0:newDim, ...]
            self.expressionPcaVar = self.expressionPcaVar[0:newDim, ...]
            self.shapeBasisSize = newDim
            self.expBasisSize = newDim
            self.albedoBasisSize = newDim

        print("loading mesh normals...")
        dic = loadDictionaryFromPickle(pathNormals)
        self.meshNormals = MeshNormals(device, self.faces, dic['vertexIndex'], dic['vertexFaceNeighbors'])

        print("loading uv parametrization...")
        self.uvParametrization = loadDictionaryFromPickle(pathUV)

        for key in self.uvParametrization:
            if key != 'uvResolution':
                self.uvParametrization[key] = torch.tensor(self.uvParametrization[key]).to(device)

        self.uvMap = self.uvParametrization['uvVertices'].to(device)

        print("loading landmarks association file...")
        self.landmarksAssociation = torch.tensor(np.loadtxt(pathLandmarks, delimiter='\t\t')[:, 1].astype(np.int64)).to(device)
        self.landmarksMask = torch.tensor(np.loadtxt(pathLandmarks, delimiter='\t\t')[:, 0].astype(np.int64)).to(device)

        print('creating sampler...')
        self.sampler = NormalSampler(self)

    def generateTextureFromAlbedo(self, albedo):
        '''
        generate diffuse and specular textures from per vertex albedo color
        :param albedo: tensor of per vertex albedo color [n, verticesNumber, 3]
        :return: generated textures [n, self.getTextureResolution(), self.getTextureResolution(), 3]
        '''
        assert (albedo.dim() == 3 and albedo.shape[-1] == self.diffuseAlbedoMean.shape[-1] and albedo.shape[-2] == self.diffuseAlbedoMean.shape[-2])
        textureSize = self.uvParametrization['uvResolution']
        halfRes = textureSize // 2
        baryCenterWeights = self.uvParametrization['uvFaces']
        oFaces = self.uvParametrization['uvMapFaces']
        uvxyMap = self.uvParametrization['uvXYMap']

        neighboors = torch.arange(self.faces.shape[-1], dtype = torch.int64, device = self.faces.device)

        texture = (baryCenterWeights[:, neighboors, None] * albedo[:, self.faces[oFaces[:, None], neighboors]]).sum(dim=-2)
        textures = torch.zeros((albedo.size(0), textureSize, textureSize, 3), dtype=torch.float32, device = self.faces.device)
        textures[:, uvxyMap[:, 0], uvxyMap[:, 1]] = texture
        textures[:, halfRes, :, :] = (textures[:, halfRes -1, :, :] + textures[:, halfRes + 1, :, :]) * 0.5
        return textures.permute(0, 2, 1, 3).flip([1])

    def getTextureResolution(self):
        '''
        return the resolution of the texture
        :return: int scalar
        '''
        return self.uvParametrization['uvResolution']

    def computeShape(self, shapeCoff, expCoff):
        '''
        compute vertices from shape and exp coeff
        :param shapeCoff: [n, self.shapeBasisSize]
        :param expCoff: [n, self.expBasisSize]
        :return: return vertices tensor [n, verticesNumber, 3]
        '''
        assert (shapeCoff.dim() == 2 and shapeCoff.shape[1] == self.shapeBasisSize)
        assert (expCoff.dim() == 2 and expCoff.shape[1] == self.expBasisSize)

        vertices = self.shapeMean + torch.einsum('ni,ijk->njk', (shapeCoff, self.shapePca)) + torch.einsum('ni,ijk->njk', (expCoff, self.expressionPca))
        return vertices

    def computeNormals(self, vertices):
        '''
        compute normals for given vertices tensor
        :param vertices: float tensor [..., 3]
        :return: float tensor [..., 3]
        '''
        assert(vertices.shape[-1] == 3)
        return self.meshNormals.computeNormals(vertices)

    def computeDiffuseAlbedo(self, diffAlbedoCoeff):
        '''
        compute diffuse albedo from coeffs
        :param diffAlbedoCoeff:  tensor [n, self.albedoBasisSize]
        :return: diffuse colors per vertex [n, verticesNumber, 3]
        '''
        assert(diffAlbedoCoeff.dim() == 2 and diffAlbedoCoeff.shape[1] == self.albedoBasisSize)

        colors = self.diffuseAlbedoMean + torch.einsum('ni,ijk->njk', (diffAlbedoCoeff, self.diffuseAlbedoPca))
        return colors

    def computeSpecularAlbedo(self, specAlbedoCoeff):
        '''
        compute specular albedo from coeffs
        :param specAlbedoCoeff: [n, self.albedoBasisSize]
        :return: specular colors per vertex [n, verticesNumber, 3]
        '''
        assert(specAlbedoCoeff.dim() == 2 and specAlbedoCoeff.shape[1] == self.albedoBasisSize)

        colors = self.specularAlbedoMean + torch.einsum('ni,ijk->njk', (specAlbedoCoeff, self.specularAlbedoPca))
        return colors

    def computeShapeAlbedo(self, shapeCoeff, expCoeff, albedoCoeff):
        '''
        compute vertices  and diffuse/specular albedo from shape, exp and albedo coeff
        :param shapeCoeff: tensor [n, self.shapeBasisSize]
        :param expCoeff: tensor [n, self.expBasisSize]
        :param albedoCoeff: tensor [n, self.albedoBasisSize]
        :return: vertices [n, verticesNumber 3], diffuse albedo [n, verticesNumber 3], specAlbedo albedo [n, verticesNumber 3]
        '''

        vertices = self.computeShape(shapeCoeff, expCoeff)
        diffAlbedo = self.computeDiffuseAlbedo(albedoCoeff)
        specAlbedo = self.computeSpecularAlbedo(albedoCoeff)
        return vertices, diffAlbedo, specAlbedo

    def sample(self, shapeNumber = 1):
        '''
        random sample shape, expression, diffuse and specular albedo coeffs
        :param shapeNumber: number of shapes to sample
        :return: shapeCoeff [n, self.shapeBasisSize], expCoeff [n, self.expBasisSize], diffCoeff [n, albedoBasisSize], specCoeff [n, self.albedoBasisSize]
        '''
        shapeCoeff = self.sampler.sample(shapeNumber, self.shapePcaVar)
        expCoeff = self.sampler.sample(shapeNumber, self.expressionPcaVar)
        diffAlbedoCoeff = self.sampler.sample(shapeNumber, self.diffuseAlbedoPcaVar)
        specAlbedoCoeff = self.sampler.sample(shapeNumber, self.specularAlbedoPcaVar)
        return shapeCoeff, expCoeff, diffAlbedoCoeff, specAlbedoCoeff


