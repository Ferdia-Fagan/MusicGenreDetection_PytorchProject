import os
import sys
root_directory = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_directory)

from src.DataPreprocessing import DataExtractor

from CNN_1D import TrainAndTestModels as CNN1D_Models_Run
from CNN_2D import TrainAndTestModels as CNN2D_Models_Run
from LSTM import TrainAndTestModels as LSTM_Models_Run

# EXTRACT THE DATA.
DataExtractor.run()
"""
After this, the data that will be generated is:

1) Raw waveform (sr=22050)

2) Raw mel spectograms (sr=22050):
    a) frame_size = 1024, hop_size = 512
    b) frame_size = 512, hop_size = 256
    c) frame_size = 256, hop_size = 128
"""

# TRAIN AND TEST MODELS:
def run():
    CNN1D_Models_Run.run()

    CNN2D_Models_Run.run()

    # LSTM_Models_Run.run()


if __name__ == '__main__':
    run()



