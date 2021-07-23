import os

from src.CNN_1D import MelSpectogram_CNN_1024_LessDropout, MelSpectogram_CNN_1024_MoreDropout, \
    MelSpectogram_CNN_1024_WIthGAP, MelSpectogram_CNN_1024_ConvsInsteadOfPools, \
    MelSpectogram_CNN_1024_ConvsInsteadOfPools_withoutPumping, MelSpectogram_CNN_512, RawAudio_CNN, \
    RawAudio_CNN_WithGAP, MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithBottleneckLayer, \
    MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithBottleneckLayer_WithGAP, RawAudio_CNN_WithGAP_WithBottlenecks, \
    RawAudio_CNN_WithGAP_WithSomeAdjustments, MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithGAP, \
    MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithGAP_WithAdjustments, \
    MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithGAP_Smaller
from src.CNN_1D.MelSpectogramExperimentalModels import Experiment, VGG_16, AlexNet
from src.utils import HParams, Main

Main.setUpReproducibility()

# root_directory = f"{os.getcwd()}"
root_directory = os.path.abspath(os.path.dirname(__file__))

def runModel(model_name_testing, modelToTest):
    model_parameters_testing = HParams.loadModelParameters(root_directory,model_name_testing)

    datasetPath = f"{root_directory}/../../{model_parameters_testing.data_dir}"

    Main.runModelWithParams(modelToTest, model_parameters_testing,
                            model_name_testing, root_directory, datasetPath)

# -----------------------------------------------------------

def run():
    model_name_testing = "experiment"
    runModel(model_name_testing, Experiment)

    model_name_testing = "VGG_16"
    runModel(model_name_testing, VGG_16)

    model_name_testing = "Alex_Net"
    runModel(model_name_testing, AlexNet)

    model_name_testing = "CNN_1D_RawMelSpectogram_1024_lessDropout"
    runModel(model_name_testing, MelSpectogram_CNN_1024_LessDropout)

    model_name_testing = "MelSpectogram_CNN_1024_MoreDropout"
    runModel(model_name_testing, MelSpectogram_CNN_1024_MoreDropout)

    model_name_testing = "MelSpectogram_CNN_1024_WithGAP"
    runModel(model_name_testing, MelSpectogram_CNN_1024_WIthGAP)

    model_name_testing = "MelSpectogram_CNN_1024_ConvsInsteadOfPools"
    runModel(model_name_testing, MelSpectogram_CNN_1024_ConvsInsteadOfPools)

    model_name_testing = "MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithGAP"   # TODO:
    runModel(model_name_testing, MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithGAP)
    #
    model_name_testing = "MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithGAP_WithAdjustments"   # TODO:
    runModel(model_name_testing, MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithGAP_WithAdjustments)

    model_name_testing = "MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithGAP_Smaller"   # TODO:
    runModel(model_name_testing, MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithGAP_Smaller)
    #
    #
    # # model_name_testing = "MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithBottleneckLayer"     # disabled because not interesting
    # # runModel(model_name_testing, MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithBottleneckLayer)
    #
    # # model_name_testing = "MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithBottleneckLayer_WithGAP"   # disabled because not interesting
    # # runModel(model_name_testing, MelSpectogram_CNN_1024_ConvsInsteadOfPools_WithBottleneckLayer_WithGAP)
    #
    model_name_testing = "MelSpectogram_CNN_1024_ConvsInsteadOfPools_withoutPumping"
    runModel(model_name_testing, MelSpectogram_CNN_1024_ConvsInsteadOfPools_withoutPumping)

    model_name_testing = "CNN_1D_RawMelSpectogram_512"
    runModel(model_name_testing, MelSpectogram_CNN_512)

    model_name_testing = "CNN_1D_RawMelSpectogram_256"
    runModel(model_name_testing, MelSpectogram_CNN_512)

    model_name_testing = "CNN_1D_RawAudio_100%"
    runModel(model_name_testing,RawAudio_CNN)
    #
    # # model_name_testing = "CNN_1D_RawAudio_100%_Better"    # this is worse
    # # runModel(model_name_testing,RawAudio_CNN)
    #
    model_name_testing = "CNN_1D_RawAudio_100%_WithGAP"
    runModel(model_name_testing,RawAudio_CNN_WithGAP)

    model_name_testing = "CNN_1D_RawAudio_100%_WithGAP_WithSomeAdjustments"     # TODO:
    runModel(model_name_testing,RawAudio_CNN_WithGAP_WithSomeAdjustments)

    # model_name_testing = "CNN_1D_RawAudio_100%_WithGAP_WithBottlenecks"   # not interesting
    # runModel(model_name_testing,RawAudio_CNN_WithGAP_WithBottlenecks)

# run()


