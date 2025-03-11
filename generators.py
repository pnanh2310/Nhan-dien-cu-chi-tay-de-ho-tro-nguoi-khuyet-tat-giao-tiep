from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_preprocessing import train_df, valid_df, test_df

train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='Filepaths', y_col='Labels',
    target_size=(150, 150), batch_size=50, class_mode='categorical', shuffle=True
)

valid_generator = valid_datagen.flow_from_dataframe(
    valid_df, x_col='Filepaths', y_col='Labels',
    target_size=(150, 150), batch_size=50, class_mode='categorical', shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    test_df, x_col='Filepaths', y_col='Labels',
    target_size=(150, 150), batch_size=50, class_mode='categorical', shuffle=False
)
