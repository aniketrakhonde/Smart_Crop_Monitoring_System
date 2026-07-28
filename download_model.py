import kagglehub

path = kagglehub.model_download(
    "mgmitesh/plant-disease-detection-model/tensorFlow2/default"
)

print(path)