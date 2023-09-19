import sys
import os
import torch
from optimizer import Optimizer
from config import Config
import glob
sys.path.insert(0,'/content/NextFace') # Verify your path

config = Config()
config.fillFromDicFile('./optimConfig.ini')
# config.device = 'cuda' # torch not compiled with CUDA
config.path = './baselMorphableModel/' # Verify your path

# Directory path containing all images
imageFolderPath = './input/test/s1.png'

outputDir = './output/test/all_images/s1_mitsuba_mat'
#setup
# Extract the folder name 
folder_name = os.path.basename(imageFolderPath.strip('/'))

outputDir = './output/' + folder_name + '/'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)  # Create the output directory if it doesn't exist
    
torch.cuda.set_device(0)

torch.cuda.empty_cache()
optimizer = Optimizer(outputDir, config)

# Run the optimization for the current image
# optimizer.run(imagePath, doStep1=True, doStep2=True, doStep3=True, renderer="vertex")
# Based on the vertex_based optimization, try to get to the same result with the Mitsuba one
# optimizer.run(outputImageDir+'/debug/results/ref.png', checkpoint=outputImageDir+'/checkpoints/stage1_output.pickle', doStep1=False, doStep2=True, doStep3=False, renderer="mitsuba")
# optimizer.run(imagePath, doStep1=True, doStep2=True, doStep3=True, rendererName="redner")
# optimizer.run(imagePath, doStep1=True, doStep2=True, doStep3=True, rendererName="mitsuba")
optimizer.run(imageFolderPath, doStep1=False, doStep2=True, doStep3=False, rendererName="mitsuba")
# optimizer.run(imagePath, checkpoint=outputImageDir+'/checkpoints/stage2_output.pickle',doStep1=False, doStep2=False, doStep3=True, renderer="mitsuba")
# optimizer.run(imagePath, doStep1=True, doStep2=True, doStep3=False, renderer="redner")