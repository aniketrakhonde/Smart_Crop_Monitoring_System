from tensorflow.keras.models import load_model

model = load_model("models/plant_disease_model.h5", compile=False)

print("Input Shape:", model.input_shape)
print("Output Shape:", model.output_shape)
print("Number of classes:", model.output_shape[-1])