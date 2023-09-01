import torch
import math
import pyredner
import redner
import random

from pyredner import set_print_timing
from renderers.renderer import Renderer 


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

class RendererRedner(Renderer):

    def __init__(self, samples, bounces, device, screenWidth, screenHeight):
        set_print_timing(False) #disable redner logs
        self.samples = samples
        self.bounces = bounces
        self.device = torch.device(device)
        self.clip_near = 10.0
        self.upVector = torch.tensor([0.0, -1.0, 0.0])
        self.counter = 0
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight

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

    def buildScenes(self, vertices, indices, normals, uv, diffuseTexture, specularTexture, roughnessTexture, focal, envMaps):
        '''
        build multiple pyredner scenes used for path tracing (uv mapping and indices are the same for all scenes)
        Args:
            vertices (B, N, 3): vertices tensor
            indices (indices number,3): indices of the morphable model
            normals (B, N, 3): normals of our vertices
            uv (N, 3): uv map
            diffuseTexture (B, resX, resY, 3) or (1, resX, resY, 3): diffuse textures for all our images
            specularTexture (B, resX, resY, 3) or (1, resX, resY, 3): specular textures for all our images
            roughnessTexture (B, resX, resY, 3) or (1, resX, resY, 3): roughness textures for all our images
            focal [B]: focals for our scenes
            envMaps (B, resX, resY, 3) : envMap textures for all our images

        Returns:
            images [B, resX, resY, 4]: the renders based on our inputs
        '''
        assert(vertices.dim() == 3 and vertices.shape[-1] == 3 and normals.dim() == 3 and normals.shape[-1] == 3)
        assert (indices.dim() == 2 and indices.shape[-1] == 3)
        assert (uv.dim() == 2 and uv.shape[-1] == 2)
        assert (diffuseTexture.dim() == 4 and diffuseTexture.shape[-1] == 3 and
                specularTexture.dim() == 4 and specularTexture.shape[-1] == 3 and
                roughnessTexture.dim() == 4 and roughnessTexture.shape[-1] == 1)
        assert(focal.dim() == 1)
        assert(envMaps.dim() == 4 and envMaps.shape[-1] == 3)
        assert(vertices.shape[0] == focal.shape[0] == envMaps.shape[0])
        assert(diffuseTexture.shape[0] == specularTexture.shape[0] == roughnessTexture.shape[0])
        assert (diffuseTexture.shape[0] == 1 or diffuseTexture.shape[0] == vertices.shape[0])
        sharedTexture = True if diffuseTexture.shape[0] == 1 else False

        scenes = []
        for i in range(vertices.shape[0]):
            texIndex = 0 if sharedTexture else i
            mat = pyredner.Material(pyredner.Texture(diffuseTexture[texIndex]),
                                    pyredner.Texture(specularTexture[texIndex]) if specularTexture is not None else None,
                                    pyredner.Texture(roughnessTexture[texIndex]) if roughnessTexture is not None else None)
            obj = pyredner.Object(vertices[i], indices, mat, uvs=uv, normals=normals[i] if normals is not None else None)
            cam =  self.setupCamera(focal[i], self.screenWidth, self.screenHeight)
            scene = pyredner.Scene(cam, materials=[mat], objects=[obj], envmap=pyredner.EnvironmentMap(envMaps[i]))
            scenes.append(scene)

        return scenes

    def renderImageAlbedo(self, scenes):
        """
        render scenes with ray tracing (only albedo)
        Args:
            scenes:  list of pyredner scenes
        Returns:
            images [B, resX, resY, 4]: albedo images
        """
        #images =pyredner.render_albedo(scenes, alpha = True, num_samples = self.samples, device = self.device)
        images = renderPathTracing(scenes,
                                   channels= [pyredner.channels.diffuse_reflectance, pyredner.channels.alpha],
                                   max_bounces = 0,
                                   num_samples = self.samples ,
                                   device = self.device)
        return images
    def renderImage(self, scenes):
        """
        render scenes with ray tracing (only albedo)
        Args:
            scenes:  list of pyredner scenes
        Returns:
            images [B, resX, resY, 4]: images
        """
        images = renderPathTracing(scenes,
                                   max_bounces = self.bounces,
                                   num_samples = self.samples ,
                                   device = self.device)
        self.counter += 1
        return images
    
    def render(self, cameraVertices, indices, normals, uv, diffAlbedo, diffuseTexture, specularTexture, roughnessTexture, shCoeffs, sphericalHarmonics, focals, renderAlbedo=False, lightingOnly=False):
        """
        Render images with redner based on inputs

        Args:
            vertices (B, N, 3): vertices tensor
            indices (indices number,3): indices of the morphable model
            normals (B, N, 3): normals of our vertices
            uv (N, 3): uv map
            diffAlbedo (B, N, 3) : diffuse albedo from coefficients
            diffuseTexture (B, resX, resY, 3) or (1, resX, resY, 3): diffuse textures for all our images
            specularTexture (B, resX, resY, 3) or (1, resX, resY, 3): specular textures for all our images
            roughnessTexture (B, resX, resY, 3) or (1, resX, resY, 3): roughness textures for all our images
            shCoeffs (B, sh order ^2,3) : sh coefficients
            sphericalHarmonics : SH class object that helps us to do envMap conversions
            focals (B): focals for our scenes
            renderAlbedo bool : render only with albedo
            lightingOnly bool : render only the lighting impact
        Returns:
            images (B, resX, resY, 4): the renders based on our inputs
        """
        envMaps = sphericalHarmonics.toEnvMap(shCoeffs)
        assert(envMaps.dim() == 4 and envMaps.shape[-1] == 3)
        assert(cameraVertices.shape[0] == envMaps.shape[0])
        
        scenes = self.buildScenes(cameraVertices, indices, normals, uv, diffuseTexture, specularTexture, torch.clamp(roughnessTexture, 1e-20, 10.0), focals, envMaps)
        if renderAlbedo:
            images = self.renderImageAlbedo(scenes)
        else:
            images = self.renderImage(scenes)
                
        return images
    
    
    