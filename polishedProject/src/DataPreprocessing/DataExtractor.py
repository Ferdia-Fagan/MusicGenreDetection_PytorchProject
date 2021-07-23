import os
import numpy as np

import torchaudio
import torch

from src.utils.HParams import HParams

root_directory = os.path.abspath(os.path.dirname(__file__))

root_data_folder = os.path.realpath(f"{root_directory}../../../Data/")


featureExtractionParametersYaml = "FeatureExtractionParameters.yaml"

class Extractor:
    """
    Processing raw audio and extracts a new data out of it.
    Currently it extracts:
        (1) Mel-spectograms
        (2) Raw audio

    Also performs data augmentation by splitting songs into equal parts
    (the default is 10 3s parts)
    """
    def __init__(self,params,squeezeTo1D):
        if params.isExtractingRawMelSpectogram:
            self.isSqueezingTo1D = squeezeTo1D

            self.extraction_destination = f"{root_data_folder}/{params.destination_folder_name}"
            self.n_fft = params.n_fft
            self.hop_length = params.hop_length
            self.win_length = params.win_length
            self.num_mels = params.num_mels
            self.new_sr = params.new_sr
            self.standardLengthPerSong = params.standardLengthPerSong

            self.splitIntoChunksNumber = params.splitIntoChunksNumber

            self.isExtractingRawMelSpectogram = params.isExtractingRawMelSpectogram
        else:
            self.extraction_destination = f"{root_data_folder}/{params.destination_folder_name}"
            self.new_sr = params.new_sr

            self.standardLengthPerSong = params.standardLengthPerSong

            self.splitIntoChunksNumber = params.splitIntoChunksNumber

            self.isExtractingRawMelSpectogram = params.isExtractingRawMelSpectogram

    def extractMelSpectrogramRawFeature(self,songFileLocation):
        waveform, sample_rate = torchaudio.load(songFileLocation)

        resample_transform = torchaudio.transforms.Resample(
           orig_freq=sample_rate, new_freq=self.new_sr)
        audio_mono = torch.mean(resample_transform(waveform),
                                dim=0, keepdim=True)

        melspectogram_transform = torchaudio.transforms.MelSpectrogram(
            n_fft=self.n_fft, win_length=self.win_length,
            hop_length=self.hop_length,n_mels=self.num_mels)
        melspectogram_db_transform = torchaudio.transforms.AmplitudeToDB()

        melspectogram = melspectogram_transform(audio_mono)
        melspectogram_db = melspectogram_db_transform(melspectogram)

        if self.isSqueezingTo1D:
            melspectogram_db = melspectogram_db.squeeze()
        melspectogram_db = melspectogram_db.numpy().T
        return self.resizeMelSpectrogram(melspectogram_db)

    def extractRawAudio(self,songFileLocation):
        waveform, sample_rate = torchaudio.load(songFileLocation)

        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.new_sr)

        audio_mono = torch.mean(resample_transform(waveform),
                                dim=0, keepdim=True).numpy().T

        return self.resizeMelSpectrogram(audio_mono)

    def resizeMelSpectrogram(self,rawMelSpectrogram):
        if self.isExtractingRawMelSpectogram:
            if self.isSqueezingTo1D:
                resizedRawMelSpectogram = np.zeros((self.standardLengthPerSong, rawMelSpectrogram.shape[0]))
            else:
                resizedRawMelSpectogram = np.zeros((self.standardLengthPerSong, rawMelSpectrogram.shape[1],rawMelSpectrogram.shape[2]))
        else:
            resizedRawMelSpectogram = np.zeros((self.standardLengthPerSong, 1))

        rawMelSpectrogram_Length = rawMelSpectrogram.shape[0]
        if rawMelSpectrogram_Length < self.standardLengthPerSong:
            # print("resample_rawMelSpectrogram : ", rawMelSpectrogram_Length)
            resizedRawMelSpectogram[:rawMelSpectrogram_Length] = rawMelSpectrogram
        else:
            # print("resizedRawMelSpectogram : ", rawMelSpectrogram_Length)
            resizedRawMelSpectogram = rawMelSpectrogram[:self.standardLengthPerSong]

        return resizedRawMelSpectogram

    def saveExtraction(self, extraction,genre,extractionName):
        if not os.path.exists(f"{self.extraction_destination}/{genre}"):  # TODO: move outside to speteate loop
            os.makedirs(f"{self.extraction_destination}/{genre}")

        if self.splitIntoChunksNumber != None:
            augmented_rawMelSpectrograms = np.split(extraction, self.splitIntoChunksNumber)
            for i, augmented_rawMelSpect in enumerate(augmented_rawMelSpectrograms):
                saveLocation = f"{self.extraction_destination}/{genre}/{extractionName}_i{i}"
                np.save(saveLocation, augmented_rawMelSpect.astype(np.float32))
        else:
            saveLocation = f"{self.extraction_destination}/{genre}/{extractionName}"
            np.save(saveLocation, extraction.astype(np.float32))

def extractDatasetExample(rootDir):
    # print("extractDatasetExample from : ", rootDir)
    for genre in next(os.walk(rootDir))[1]:
        # print("genre: ", genre)
        for songFileName in os.listdir(os.path.join(rootDir,f"{genre}")):
            songFileLoc = os.path.join(rootDir,f"{genre}/{songFileName}")
            yield songFileLoc,songFileName,genre

def extractionAction(extractionToPerform,params,squeezeTo1D):
    extractor = Extractor(params,squeezeTo1D)

    isExtractingRawMelSpectogram = params.isExtractingRawMelSpectogram
    # root_location_for_features = f"{root_data_folder}/{folder_name}"
    # print("root_data_folder: ", root_data_folder)
    # # extractDatasetExample(f"{root_data_folder}/genres_original/")
    print("root_data_folder: ", root_data_folder)
    for songFileLoc, songFileName, genre in extractDatasetExample(f"{root_data_folder}/genres_original"):
        try:
            if isExtractingRawMelSpectogram:
                resample_rawMelSpectrogram = extractor.extractMelSpectrogramRawFeature(songFileLoc)
            else:
                resample_rawMelSpectrogram = extractor.extractRawAudio(songFileLoc)
            songFileName = songFileName.replace('.wav', '.npy')

            extractor.saveExtraction(resample_rawMelSpectrogram, genre, songFileName)
            # print("the type: ", resample_rawMelSpectrogram.shape)
        except:
            print("ignoreing file as cant read it: ", songFileLoc)
    print("done extraction: ", extractionToPerform)


extractionsToPerform = ["extract_RawResampledAudio",
                        "extract_RawMelSpectrograms_augmented_1024",
                        "extract_RawMelSpectrograms_augmented_512",
                        "extract_RawMelSpectrograms_augmented_256"]


def run():
    for extractionToPerform in extractionsToPerform:
        print("start extraction: ", extractionToPerform)
        params = HParams(f"{root_directory}/FeatureExtractionParameters.yaml", extractionToPerform)
        extractionAction(extractionToPerform,params, True)


# extractionsToPerform = ["extract_RawMelSpectrograms_augmented","extract_RawMelSpectrograms_DifferentParams_augmented",
#                         "extract_RawMelSpectrograms_DifferentParams2_augmented"]

    # "extract_RawMelSpectrograms_DifferentParams2"]

    # "extract_RawMelSpectrograms_augmented","extract_RawMelSpectrograms_DifferentParams_augmented",
    #                     "extract_RawMelSpectrograms_DifferentParams2_augmented"]
    # "extract_RawMelSpectrograms_DifferentParams2"]
    # "extract_RawMelSpectrograms_DifferentParams",
                        #
#"extract_RawMelSpectrograms_DifferentParams2_augmented","extract_RawMelSpectrograms", "extract_RawMelSpectrograms_DifferentParams2"
                        # ]

# def main():
#     # params = HParams("./FeatureExtractionParameters.yaml", "extract_RawResampledAudio")
#     # extractionAction("extract_RawResampledAudio", params, True)
#     for extractionToPerform in extractionsToPerform:
#         print("start extraction: ", extractionToPerform)
#         params = HParams(featureExtractionParametersYaml, extractionToPerform)
#         extractionAction(extractionToPerform,params, True)
#

if __name__ == '__main__':
    run()



