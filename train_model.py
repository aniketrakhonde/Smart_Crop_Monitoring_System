import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os


train_path = "dataset/PlantVillage/train"
val_path = "dataset/PlantVillage/val"


img_size = 224
batch_size = 32


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)


train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    val_path,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode="categorical"
)


print("Classes:", train_data.num_classes)


base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)

output = Dense(
    train_data.num_classes,
    activation="softmax"
)(x)


model = Model(
    inputs=base_model.input,
    outputs=output
)


model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)


os.makedirs("models", exist_ok=True)

model.save(
    "models/plant_disease_model.h5"
)


print("MODEL SAVED SUCCESSFULLY")