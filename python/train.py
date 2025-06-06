import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model.emotion_model import create_emotion_model

# Đường dẫn dữ liệu
train_dir = r'C:\emotion-vision\data\archive (2)\train'
test_dir = r'C:\emotion-vision\data\archive (2)\test'

# Tạo ImageDataGenerator với augmentation cho train, chỉ rescale cho test
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,        # xoay ảnh +/-15 độ
    zoom_range=0.15,          # zoom +/-15%
    width_shift_range=0.1,    # dịch ngang 10%
    height_shift_range=0.1,   # dịch dọc 10%
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load dữ liệu, resize ảnh về 48x48 grayscale
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

# Tạo model
model = create_emotion_model(input_shape=(48,48,1), num_classes=train_data.num_classes)

# Compile model với Adam optimizer
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback dừng sớm khi val_loss không cải thiện trong 4 epoch
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Callback giảm learning rate khi val_loss không cải thiện trong 2 epoch
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Huấn luyện model
model.fit(
    train_data,
    validation_data=test_data,
    epochs=30,              # tăng epochs lên để model có thể học sâu hơn
    callbacks=[early_stop, reduce_lr]
)

# Lưu model
model.save('emotion_model.h5')
print("✅ Model đã được lưu thành công: emotion_model.h5")
