# NextFace
NextFace is a light-weight pytorch library for high-fidelity 3D face reconstruction from monocular image(s) where scene attributes –3D geometry, reflectance (diffuse, specular and roughness), pose, camera parameters, and scene illumination– are estimated. It is first-order optimization method that uses pytorch autograd engine and ray tracing to fit a statistical morphable model to an input image(s).
<p align="center">
<img src="resources/visual.png" >
</p>

# Features: 
* Reconstructs face at high fidelity from single or mutliple RGB images
* Estimates face geometry 
* Estimates detailed face refelctance (diffuse, specular and roughness) 
* Estimates scene light with spherical harmonics
* Estimates head pose and orientation
* Runs on both cpu and   cuda-enabled gpu


# Installation
* Clone the repository 
* Create a new: conda env create -f environment.yml
* Activate the environment: conda activate nextFace
* Download basel face model from here (https://faces.dmi.unibas.ch/bfm/bfm2017.html), just fill the form and you will receive an instant direct download link in your inbox. Downloaded  model2017-1_face12_nomouth.h5 file and put it inside ./baselMorphableModel 


# How to use
* to reconstruct a face from a single image: run the following command:
	* **python optimizer.py --input *path-to-your-input-image* --output *output-path-where-to-save-results***
* In case you have multiple images with same resolution, u can run a batch optimization on these images. For this, put all ur images in the same directory and run the following command: 
	 * **python optimizer.py --input *path-to-your-folder-that-contains-all-ur-images* --output *output-path-where-to-save-results***
* if you have multiple images for the same person u can run the following command:
	 * **python optimizer.py --sharedIdentity --input *path-to-your-folder-that-contains-all-ur-images* --output *output-path-where-to-save-results***

	the **sharedIdentity** flag tells the optimizer that all images belong to the same person. in such case the shape identity and face reflectance attributes are shared across all images. this generally produce  better face reflectance and geometry estimation. 
	
* The file **optimConfig.ini** allows to control different aspect of NextFace such as:
	* optimization (regularizations, number of iterations...)
	* compute device (run on cpu or gpu)
	* spherical harmonics (number of bands, environment map resolution)
	* ray tracing (number of samples)
* The code is self-documented and easy to follow 

# Output 
The optimization takes 4~5 minutes depending on your gpu performance. The output of the optimization is the following:
* render_{imageIndex}.png: contains from left to right: input image, overlay of the final reconstruction on the input image, the final reconstruction, diffuse, specular and roughness maps projected on the face. 
* diffuseMap_{imageIndex}.png: the estimated diffuse map in uv space
* specularMap_{imageIndex}.png: the estimated specular map in uv space
* roughnessMap_{imageIndex}.png: the estimated roughness map in uv space
* mesh{imageIndex}.obj: an obj file that contains the 3D mesh of the reconstructed face

# How it works 
NextFace reprocudes the optimizatin strategy of our [early work](https://arxiv.org/abs/2101.05356). The optimization is composed of the three stages:
* **stage 1**: or coarse stage, where face expression and head pose are estimated by minimizing the landmarks loss between the 2d landmarks and their corresponding face vertices. this produces a good starting point for the next optimization stage
* **stage 2**: the face shape identity/expression,  statistical diffuse and specular albedos, head pose and scene light are estimated by minimizing the photo consistency loss between the ray traced image and the real one.
* **stage 3**: to improve the statistical albedos estimated in the previous stage, the method optimizes, on per-pixel basis, the previously estimated albedos and try to capture more albedo details. Consistency, symmetry and smoothness regularizers (similar to [this work](https://arxiv.org/abs/2101.05356)) are used to avoid overfitting and add robustness against lighting conditions.  
By default,  the method uses 9 order spherical harmonics bands (as in [this work](https://openaccess.thecvf.com/content/ICCV2021/papers/Dib_Towards_High_Fidelity_Monocular_Face_Reconstruction_With_Rich_Reflectance_Using_ICCV_2021_paper.pdf)) to capture scene light. you can modify the number of spherical harmonics bands  in **optimConfig.ini** bands and see the importance of using high number of bands for a better light recovery. 
# Good practice for best reconstruction

* To obtain best reconstruction with optimal albedos, ensure that the images are taken in good lighting conditions (no shadows and well lit...).
* In case of single input image, ensure that the face is frontal to reconstructs a complete diffuse/specular/roughness, as the method recover only visible parts of the face. 
* Avoid extreme face expressions as the underlying model may fail to recover them. 
# Limitations 
* The method relies on detected landmarks to initialize the optimization. In case these landmarks are incorrect, you may get sub-optimal reconstruction. NextFace uses landmarks from [face_alignment](https://github.com/1adrianb/face-alignment) which is robust against extreme poses however it is not as accurate as it can be. This limitation has been discussed [here](https://openaccess.thecvf.com/content/ICCV2021/papers/Dib_Towards_High_Fidelity_Monocular_Face_Reconstruction_With_Rich_Reflectance_Using_ICCV_2021_paper.pdf) and [here](https://arxiv.org/abs/2101.05356). Using [this landmark detector](https://arxiv.org/abs/2204.02776) from Microsoft seems promising. 
* NextFace is slow and execution speed decreases with the size of the input image. For instance, if you are running an old-gpu (like me), you can decrease the resolution of the input image in the **optimConfig.ini** file by reducing the value of the *maxResolution* parameter. Our [recent work](https://openaccess.thecvf.com/content/ICCV2021/papers/Dib_Towards_High_Fidelity_Monocular_Face_Reconstruction_With_Rich_Reflectance_Using_ICCV_2021_paper.pdf) solves for this and achieve near real-time performance using deep convolutional neural network.
* NextFace cannot capture fine geometry details (wrinkles, pores...). these details may get baked in the final albedos. work such as [this](https://openaccess.thecvf.com/content_CVPR_2020/papers/Abrevaya_Cross-Modal_Deep_Face_Normals_With_Deactivable_Skip_Connections_CVPR_2020_paper.pdf) and [this](https://arxiv.org/abs/2203.07732) are interesting. 
* The spherical harmonics can only model lights at infinity, under strong directional shadows, the estimated light may not be accurate as it can be, so residual shadows may appear in the estimated albedos. You can attenuate this by increasing the value of regularizers in the **optimConfig.ini** file, but this trade-off albedo details. 
Below are the values to modify: 
	* for diffuse map: *weightDiffuseSymmetryReg* and *weightDiffuseConsistencyReg*, 
	* for specular map: *weightSpecularSymmetryReg*, *weightSpecularConsistencyReg*
	* for roughness map: *weightRoughnessSymmetryReg* and *weightRoughnessConsistencyReg*
I also provided a configuration file named **optimConfigLight.ini** which have higher regularization values for these maps that u can use
* Using a single image to estimate face attribute is an ill-posed problem and the estimated reflectance maps(diffuse, specular and roughness) are view/camera dependent. if you to obtain more intrinsic reflectance maps, you have to use multiple images per subject.

# Roadmap
If I have time:
* Add virtual lightstage as proposed in [this]() to model high frequency point lights.
* Add support for [FLAME](https://github.com/Rubikplayer/flame-fitting) morphable model. You are welcome if you can help. 
* add GUI interface for loading images, landmarks edition, run optimization and visualize results.
 
# License
NextFace is available for free, under GPL license, to use for research and educational purposes only. Please check LICENSE file.

# Acknowledgements
The uvmap is taken from [here](https://github.com/unibas-gravis/parametric-face-image-generator/blob/master/data/regions/face12.json), landmarks association  from [here](https://github.com/kimoktm/Face2face/blob/master/data/custom_mapping.txt). [redner](https://github.com/BachiLi/redner/) is used for ray tracing, albedo model from [here](https://github.com/waps101/AlbedoMM/).

# contact 
mail: deeb.abdallah @at gmail
twitter: abdallah_dib

# Citation 
If you use NextFace and find it useful in your work, plz cite the following work:

```
@inproceedings{dib2021practical,
  title={Practical face reconstruction via differentiable ray tracing},
  author={Dib, Abdallah and Bharaj, Gaurav and Ahn, Junghyun and Th{\'e}bault, C{\'e}dric and Gosselin, Philippe and Romeo, Marco and Chevallier, Louis},
  booktitle={Computer Graphics Forum},
  volume={40},
  number={2},
  pages={153--164},
  year={2021},
  organization={Wiley Online Library}
}
