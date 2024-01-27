from pathlib import Path, PurePath
import os

class ModelDefenition:
    model = ""
    config = ""
    cluster = None

    speaker = 0

    transpose = 0
    auto_predict_f0 = False
    f0_method = "dio"

    cluster_infer_ratio = 0
        
    noise_scale = 0.4
    db_thresh = -40

    pad_seconds = 0.5
    chunk_seconds = 0.5
    absolute_thresh = False
    max_chunk_seconds = 40
    def __init__(self, modelDict):
        if modelDict["autodetectModel"]:
            modelDirectory = Path(modelDict["modelDirLocation"])
            modelName =  Path(modelDict["modelName"])
            modelPath = PurePath(modelDirectory, modelName)
            self.model = modelDict["model"] if "model" in modelDict else Path()
            for filename in os.listdir(modelPath):
                if filename.startswith("G_"):
                    self.model = Path(modelPath, filename)

            self.cluster = modelDict["cluster"] if "cluster" in modelDict else None
            for filename in os.listdir(modelPath):
                if filename.startswith("kmeans"):
                    self.cluster = Path(modelPath, filename)

            self.config = modelDict["config"] if "model" in modelDict else Path(modelPath, "config.json")
        else:
            self.model = Path(modelDict["model"])
            self.cluster = Path(modelDict["cluster"]) if "cluster" in modelDict else None
            self.config = Path(modelDict["config"])
        self.transpose = modelDict["transpose"]
        self.auto_predict_f0 = modelDict["auto_predict_f0"]
        self.f0_method = modelDict["f0_method"]

        self.cluster_infer_ratio = modelDict["cluster_infer_ratio"]

        self.noise_scale = modelDict["noise_scale"]
        self.db_thresh = modelDict["db_thresh"]

        self.pad_seconds = modelDict["pad_seconds"]
        self.chunk_seconds = modelDict["chunk_seconds"]
        self.absolute_thresh = modelDict["absolute_thresh"]
        self.max_chunk_seconds = modelDict["max_chunk_seconds"]

        self.speaker = modelDict["speaker"]