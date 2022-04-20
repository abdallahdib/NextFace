from _nsis import out

from optimizer import Optimizer
from config import Config
from utils import *
import math
from image import saveImage

frameIndex = 0
outputDir = './out'
def produce(optimizer):
    global frameIndex
    images = optimizer.pipeline.render(None, optimizer.vEnhancedDiffuse, optimizer.vEnhancedSpecular, optimizer.vEnhancedRoughness)
    for i in range(images.shape[0]):
        fileName = outputDir + '/frame_' + str(i)+ '_%04d.png' % frameIndex
        saveImage(images[i], fileName)

    frameIndex += 1


if __name__ == "__main__":

    '''
    this code is used to rotate on vertical axis, an existing reconstruction from a pickle file. 
    u need to have ffmpeg to produce the final gif image or video 
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="path to a pickle file that contains the reconstruction (check optimizer.py)")

    parser.add_argument("--output", required=True,
                        help="path to where to save the animation sequence.")

    parser.add_argument("--config", required=False, default='./optimConfig.ini',
                        help="path to the configuration file (used to configure the optimization)")

    params = parser.parse_args()

    config = Config()

    configFile = params.config # './optimConfig.ini'
    outputDir = params.output + '/' #'./replay/'
    parameters = params.input #'../workspace/exp/checkpoints/stage3_output.pickle'

    mkdir_p(outputDir)
    config.fillFromDicFile(configFile)
    optimizer = Optimizer(outputDir, config)
    optimizer.pipeline.renderer.samples = config.rtSamples
    optimizer.loadParameters(parameters)

    DTR = math.pi / 180.0
    minBound = -30.0 * DTR  # -65
    maxBound = 30.0 * DTR  # 65
    step = 2.85 * DTR  # 0.75

    initAngle = optimizer.pipeline.vRotation[..., 1].clone()
    currentAngle = initAngle.clone()

    if True:
        frameIndex = 0
        print('animating reconstruction, this may take some time depending on the number of raytracing samples and ur gpu. please wait...')
        while currentAngle > minBound:
            currentAngle -= step
            optimizer.pipeline.vRotation[..., 1] = currentAngle
            produce(optimizer)

        while currentAngle < maxBound:
            currentAngle += step
            optimizer.pipeline.vRotation[..., 1] = currentAngle
            produce(optimizer)

        while currentAngle > initAngle:
            currentAngle -= step
            optimizer.pipeline.vRotation[..., 1] = currentAngle
            produce(optimizer)

        optimizer.pipeline.vRotation[..., 1] = initAngle
        produce(optimizer)

    import os

    #cmd = "ffmpeg -y -i " + outputDir + "frame_0_%04d.png -vf fps=25 -vcodec png -pix_fmt rgba " + outputDir + "/optimized.mov"
    cmd = "ffmpeg -f image2 -framerate 20 -y -i " + outputDir + "frame_0_%04d.png " + outputDir + "/optimized.gif"
    os.system(cmd)





