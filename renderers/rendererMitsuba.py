import drjit as dr
import mitsuba as mi
import torch
from renderers.renderer import Renderer 
from mitsuba.scalar_rgb import Transform4f as T

mi.set_variant('cuda_ad_rgb')
# mi.set_log_level(mi.LogLevel.Debug)
# mi.DEBUG = True
# dr.set_flag(dr.JitFlag.VCallRecord, False)
# dr.set_flag(dr.JitFlag.LoopRecord, False)
from plugins import *
class RendererMitsuba(Renderer):

    def __init__(self, samples, bounces, device, screenWidth, screenHeight):
        self.samples = samples
        self.bounces = bounces
        assert(device == 'cuda') # 'we only support GPU computation on mitsuba'
        self.device = torch.device(device)
        self.counter = 0
        self.screenWidth = screenWidth
        self.screenHeight = screenHeight
        dr.set_device(0)
        self.scene = self.buildInitialScene() # init my scene
        
    def buildInitialScene(self):
        """
        generate a placeholder scene where we assign default values that will be updated at runtime
        ** make sure the mitsuba_default folder ** is properly placed in the output folder
        Returns:
            scene (python dictionnary): mitsuba scene object
        """
        # generate blank textures
        # [resX, resY, 3]
        blankTexture = torch.empty((self.screenWidth, self.screenHeight, 3), device=self.device)
        bitmapTexture = mi.util.convert_to_bitmap(blankTexture)
        m_bsdf = mi.load_dict({
                    'type': 'rednermat',
                    'albedo': {
                        'type': 'bitmap',
                        'bitmap': bitmapTexture
                    },                
                    'roughness':{
                        'type': 'bitmap',
                        'bitmap': bitmapTexture
                        
                    },
                    'specular':{
                        'type': 'bitmap',
                        'bitmap': bitmapTexture
                        
                    }
                })
        # Create scene
        self.scene = mi.load_dict({
            'type': 'scene',
            'integrator': {
                'type':'aov',
                'aovs':'dd.y:depth',
                'my_image':{
                    'type':'direct_reparam'
                }
            },
            'sensor':  {
                'type': 'perspective',
                'fov': 5,
                'to_world': T.look_at(
                            origin=(0, 0, 0),
                            target=(0, 0, 1),
                            up=(0., -1, 0.)
                        ),
                'film': {
                    'type': 'hdrfilm',
                    'filter':{
                        'type':'box'
                    },
                    'width':  self.screenWidth,
                    'height': self.screenHeight,
                    'pixel_format':'rgba',
                    'sample_border':True,
                },
            },
            "mesh":{
                "type": "obj",
                "filename": "./output/mitsuba_default/mesh0.obj",
                "face_normals": True,
                'bsdf': m_bsdf
            },
            'light': {
                'type': 'envmap',
                'bitmap': bitmapTexture
            }
        })
        return self.scene
    
    # STANDALONE because of wrap_ad
    @dr.wrap_ad(source='torch', target='drjit')
    def render_torch_djit(scene, vertices, indices, normals, uv, diffuseTexture, specularTexture, roughnessTexture, fov, envMaps, spp=8, seed=1):
        """
        Standalone function that converts torch computations in mitsuba's drjit computations.
        This wrapper handles the propagation of the gradients directly. 
        We also update the values of our mitsuba scene with the values passed as inputs
        Args : 
            scene : Mitsuba scene object
            vertices (B, N, 3): vertices tensor
            indices (indices number,3): indices of the morphable model
            normals (B, N, 3): normals of our vertices
            uv (N, 3): uv map
            diffuseTexture (B, resX, resY, 3) or (1, resX, resY, 3): diffuse textures for all our images
            specularTexture (B, resX, resY, 3) or (1, resX, resY, 3): specular textures for all our images
            roughnessTexture (B, resX, resY, 3) or (1, resX, resY, 3): roughness textures for all our images
            fov (B) : field of view
            envMaps (B, resX, resY, 3) : environment map for the scene based on the shCoeffs
            spp int : number of samples
            seed int : seed number
        Returns:
           image [B, resX, resY, 9]: image with 9 channels (3 rgb + 5 depth)
        """
        params = mi.traverse(scene)
        params["sensor.x_fov"] = fov
        # update mesh params
        params["mesh.vertex_positions"] = dr.ravel(mi.TensorXf(vertices))
        params["mesh.faces"] = dr.ravel(mi.TensorXf(indices))
        params["mesh.vertex_normals"] = dr.ravel(mi.TensorXf(normals))
        params["mesh.vertex_texcoords"] = dr.ravel(mi.TensorXf(uv))
        # update BSDF
        # https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#smooth-diffuse-material-diffuse
        # RednerMat
        params["mesh.bsdf.albedo.data"] = mi.TensorXf(dr.clip(diffuseTexture, 0, 1))
        params["mesh.bsdf.specular.data"] = mi.TensorXf(dr.clip(specularTexture, 0, 1)) 
        params["mesh.bsdf.roughness.data"] = mi.TensorXf(dr.clip(roughnessTexture, 0.00001, 10.0))
        #update envMaps
        params["light.data"] = mi.TensorXf(envMaps)
        params.update() 
        img = mi.render(scene, params, spp=spp, seed=seed, seed_grad=seed+1)
        return img
        
    # overloading this method
    
    def render(self, cameraVertices, indices, normals, uv, diffAlbedo, diffuseTexture, specularTexture, roughnessTexture, shCoeffs, sphericalHarmonics, focals, renderAlbedo=False, lightingOnly=False):
        """
        middle function between pytorch and mitsuba, we take the tensor values from our pipeline and give it to our standalone wrapper

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
        envMap = sphericalHarmonics.toEnvMap(shCoeffs)
        assert(envMap.dim() == 4 and envMap.shape[-1] == 3)
        assert(cameraVertices.shape[0] == envMap.shape[0])
        
        self.fov =  torch.tensor([360.0 * torch.atan(self.screenWidth / (2.0 * focals)) / torch.pi]) # from renderer.py
        
        img =  RendererMitsuba.render_torch_djit(self.scene, cameraVertices.squeeze(0), indices.to(torch.float32), normals.squeeze(0), uv, diffuseTexture.squeeze(0), specularTexture.squeeze(0), roughnessTexture.squeeze(0), self.fov.item(), envMap.squeeze(0),self.samples) # returns a pytorch
        rgb_channels = img[..., :3]
        #debug alpha
        mask_alpha = img[..., 4:]  # only take the last channel ?
        # Create a binary mask based on a condition (e.g., depth_mean > threshold)
        threshold = 0.9 # Adjust the threshold as needed
        depth_mean = torch.mean(mask_alpha, axis=-1, keepdim=True)
        mask_alpha = (depth_mean > threshold).float()
        # Concatenate the RGB channels with the binary mask to create the final image 
        final_image = torch.cat((rgb_channels, mask_alpha), dim=-1).unsqueeze(0)

        return final_image 
        