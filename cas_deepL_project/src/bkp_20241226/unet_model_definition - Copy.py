from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    Cropping2D,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model

def unet_mode_detect_greenery(input_size=(513, 513, 3), num_classes=2):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = Conv2D(128, 3, activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation="relu", padding="same")(pool2)
    conv3 = Conv2D(256, 3, activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation="relu", padding="same")(pool3)
    conv4 = Conv2D(512, 3, activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = Conv2D(1024, 3, activation="relu", padding="same")(pool4)
    conv5 = Conv2D(1024, 3, activation="relu", padding="same")(conv5)

    # Decoder
    up6 = UpSampling2D(size=(2, 2))(conv5)
    conv4_shape = conv4.shape[1:3]
    up6_shape = up6.shape[1:3]
    crop_h = max(0, conv4_shape[0] - up6_shape[0])
    crop_w = max(0, conv4_shape[1] - up6_shape[1])
    cropped_conv4 = Cropping2D(((0, crop_h), (0, crop_w)))(conv4)
    up6 = Concatenate()([up6, cropped_conv4])
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(up6)
    conv6 = Conv2D(512, 3, activation="relu", padding="same")(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Concatenate()([up7, conv3])
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(up7)
    conv7 = Conv2D(256, 3, activation="relu", padding="same")(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Concatenate()([up8, conv2])
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(up8)
    conv8 = Conv2D(128, 3, activation="relu", padding="same")(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    conv1_shape = conv1.shape[1:3]
    up9_shape = up9.shape[1:3]
    crop_h = max(0, conv1_shape[0] - up9_shape[0])
    crop_w = max(0, conv1_shape[1] - up9_shape[1])
    cropped_conv1 = Cropping2D(((0, crop_h), (0, crop_w)))(conv1)
    up9 = Concatenate()([up9, cropped_conv1])
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(up9)
    conv9 = Conv2D(64, 3, activation="relu", padding="same")(conv9)

    # Output
    outputs = Conv2D(num_classes, 1, activation="sigmoid" if num_classes == 1 else "softmax")(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
