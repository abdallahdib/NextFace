import torch
import math
import pyredner
import redner
import random

from pyredner import set_print_timing


def rayTrace(scene,
             channels,
             max_bounces = 1,
             sampler_type = pyredner.sampler_type.sobol,
             num_samples = 8,
             seed = None,
             sample_pixel_center = False,
             device  = None):
    if device is None:
        device = pyredner.get_device()

    assert(isinstance(scene, list))
    if seed == None:
        # Randomly generate a list of seed
        seed = []
        for i in range(len(scene)):
            seed.append(random.randint(0, 16777216))
    assert(len(seed) == len(scene))
    # Render each scene in the batch and stack them together
    imgs = []
    for sc, se in zip(scene, seed):
        scene_args = pyredner.RenderFunction.serialize_scene(\
            scene = sc,
            num_samples = num_samples,
            max_bounces = max_bounces,
            sampler_type = sampler_type,
            channels = channels,
            use_primary_edge_sampling=False,
            use_secondary_edge_sampling=False,
            sample_pixel_center = sample_pixel_center,
            device = device)
        imgs.append(pyredner.RenderFunction.apply(se, *scene_args))
    imgs = torch.stack(imgs)
    return imgs

def renderPathTracing(scene,
                      channels= None,
                      max_bounces = 1,
                      num_samples = 8,
                      device = None):
    if channels is None:
        channels = [redner.channels.radiance]
        channels.append(redner.channels.alpha)
    #if alpha:
    #   channels.append(redner.channels.alpha)
    return rayTrace(scene=scene,
                    channels=channels,
                    max_bounces=max_bounces,
                    sampler_type=pyredner.sampler_type.independent,
                    num_samples=num_samples,
                    seed = None,
                    sample_pixel_center=False,
                    device=device)

class Renderer:

    def __init__(self, samples, bounces, device):
        set_print_timing(False) #disable redner logs
        self.samples = samples
        self.bounces = bounces
        self.device = torch.device(device)
        self.clip_near = 10.0
        self.upVector = torch.tensor([0.0, -1.0, 0.0])
        self.counter = 0
        self.screenWidth = 256
        self.screenHeight = 256

    def setupCamera(self, focal, image_width, image_height):

        fov = torch.tensor([360.0 * math.atan(image_width / (2.0 * focal)) / math.pi])  # calculate camera field of view from image size

        cam = pyredner.Camera(
            position = torch.tensor([0.0, 0.0, 0.0]),
            look_at = torch.tensor([0.0, 0.0, 1.0]),
            up = self.upVector,
            fov = fov.cpu(),
            clip_near = self.clip_near,
            cam_to_world = None ,
            resolution = (image_height, image_width))

        return cam

    def buildScenes(self, vertices, indices, normal, uv, diffuse, specular, roughness, focal, envMap):
        '''
        build multiple pyredner scenes used for path tracing (uv mapping and indices are the same for all scenes)
        :param vertices: [n, verticesNumber, 3]
        :param indices: [indicesNumber, 3]
        :param normal: [n, verticesNumber, 3]
        :param uv: [verticesNumber, 2]
        :param diffuse: [n, resX, resY, 3] or [1, resX, resY, 3]
        :param specular: [n, resX, resY, 3] or [1, resX, resY, 3]
        :param roughness: [n, resX, resY, 1] or [1, resX, resY, 3]
        :param focal: [n]
        :param envMap: [n, resX, resY, 3]
        :return: return list of pyredner scenes
        '''
        assert(vertices.dim() == 3 and vertices.shape[-1] == 3 and normal.dim() == 3 and normal.shape[-1] == 3)
        assert (indices.dim() == 2 and indices.shape[-1] == 3)
        assert (uv.dim() == 2 and uv.shape[-1] == 2)
        assert (diffuse.dim() == 4 and diffuse.shape[-1] == 3 and
                specular.dim() == 4 and specular.shape[-1] == 3 and
                roughness.dim() == 4 and roughness.shape[-1] == 1)
        assert(focal.dim() == 1)
        assert(envMap.dim() == 4 and envMap.shape[-1] == 3)
        assert(vertices.shape[0] == focal.shape[0] == envMap.shape[0])
        assert(diffuse.shape[0] == specular.shape[0] == roughness.shape[0])
        assert (diffuse.shape[0] == 1 or diffuse.shape[0] == vertices.shape[0])
        sharedTexture = True if diffuse.shape[0] == 1 else False

        scenes = []
        for i in range(vertices.shape[0]):
            texIndex = 0 if sharedTexture else i
            mat = pyredner.Material(pyredner.Texture(diffuse[texIndex]),
                                    pyredner.Texture(specular[texIndex]) if specular is not None else None,
                                    pyredner.Texture(roughness[texIndex]) if roughness is not None else None)
            obj = pyredner.Object(vertices[i], indices, mat, uvs=uv, normals=normal[i] if normal is not None else None)
            cam =  self.setupCamera(focal[i], self.screenWidth, self.screenHeight)
            scene = pyredner.Scene(cam, materials=[mat], objects=[obj], envmap=pyredner.EnvironmentMap(envMap[i]))
            scenes.append(scene)

        return scenes

    def renderAlbedo(self, scenes):
        '''
        render albedo of given pyredner scenes
        :param scenes:  list of pyredner scenes
        :return: albedo images [n, screenWidth, screenHeight, 4]
        '''
        #images =pyredner.render_albedo(scenes, alpha = True, num_samples = self.samples, device = self.device)
        images = renderPathTracing(scenes,
                                   channels= [pyredner.channels.diffuse_reflectance, pyredner.channels.alpha],
                                   max_bounces = 0,
                                   num_samples = self.samples ,
                                   device = self.device)
        return images

    def render(self, scenes):
        '''
        render scenes with ray tracing
        :param scenes:  list of pyredner scenes
        :return: ray traced images [n, screenWidth, screenHeight, 4]
        '''
        images = renderPathTracing(scenes,
                                   max_bounces = self.bounces,
                                   num_samples = self.samples ,
                                   device = self.device)
        self.counter += 1
        return images