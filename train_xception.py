import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set dataset paths
train_dir = r"C:\DeepFake_Defenders\dataset\HFFD_299\train"
test_dir = r"C:\DeepFake_Defenders\dataset\HFFD_299\test"

# Ensure paths exist
if not os.path.exists(train_dir):
    raise OSError(f"Train directory not found: {train_dir}")

if not os.path.exists(test_dir):
    raise OSError(f"Test directory not found: {test_dir}")

# Image size and batch size
IMG_SIZE = (299, 299)
BATCH_SIZE = 32

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Load Pretrained Xception Model
base_model = Xception(weights="imagenet", include_top=False, input_shape=(299, 299, 3))

# Freeze base model layers
base_model.trainable = False

# Add custom layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation="relu")(x)
x = Dense(512, activation="relu")(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=x)

# Compile model (No need to force GPU)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Callbacks: Save checkpoint after every epoch
checkpoint_callback = ModelCheckpoint(
    "xception_epoch_{epoch:02d}.h5",
    monitor="val_accuracy",
    save_best_only=False,
    verbose=1
)

# Early stopping to prevent overfitting
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, verbose=1, restore_best_weights=True)

# Train the Model
model.fit(
    train_generator,
    epochs=10,  
    validation_data=test_generator,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Save Final Model
model.save("xception_final_model.h5")

print("Training Completed! Model saved as xception_final_model.h5")
