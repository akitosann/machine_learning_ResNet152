import glob
from PIL import Image, ImageFile
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping
import numpy as np
import tensorflow as tf
from custom_layer import CustomLayer

# IOErrorの回避
ImageFile.LOAD_TRUNCATED_IMAGES = True

class_names = ["pal", "poke"] # 分類するクラス
num_classes = len(class_names)  # クラスの数
image_size = 224                # 画像サイズ


class ProgressCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"開始 エポック {epoch+1}/{self.params['epochs']}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"終了 エポック {epoch+1}/{self.params['epochs']}")
        print(f"損失: {logs['loss']:.4f}, 精度: {logs['accuracy']:.4f}")


# ResNet152モデル
def build_and_train_model(train_data_gen, test_data_gen, train_steps, test_steps):
    # ResNet152モデルのロード（事前訓練済みの重みを使用）
    base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))


    # カスタム層の追加
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global Average Pooling
    x = CustomLayer(my_param=10)(x)  # カスタムレイヤーの追加
    x = Dense(2048, activation='relu')(x)  # 密な接続層
    predictions = Dense(num_classes, activation='softmax')(x)  # 出力層

    # モデルの定義
    model = Model(inputs=base_model.input, outputs=predictions)

    # 事前訓練された層は訓練しないように設定
    for layer in base_model.layers:
        layer.trainable = False

    # ここでコールバックを初期化
    progress_callback = ProgressCallback()
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

    # データ拡張設定
    data_augmentation = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # ResNetに必要な前処理
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    # トップレイヤーのみを訓練
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data_gen, steps_per_epoch=train_steps, epochs=30, validation_data=test_data_gen, validation_steps=test_steps, callbacks=[progress_callback, early_stopping_callback])

    for layer in base_model.layers[-15:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data_gen, steps_per_epoch=train_steps, epochs=30, validation_data=test_data_gen, validation_steps=test_steps, callbacks=[progress_callback, early_stopping_callback])

    # モデルの保存
    model.save('./resnet152_model.h5')
    return model

# メイン実行関数
def main():
    # GPUの設定
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    data_augmentation = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_data_gen = data_augmentation.flow_from_directory(
        'path_to_train_data',
        target_size=(image_size, image_size),
        batch_size=32,
        class_mode='categorical',
        classes=class_names  # クラスの順序を明示的に指定
    )

    test_data_gen = data_augmentation.flow_from_directory(
        'path_to_test_data',
        target_size=(image_size, image_size),
        batch_size=32,
        class_mode='categorical',
        classes=class_names  # クラスの順序を明示的に指定
    )

    train_steps = np.ceil(train_data_gen.samples / 32)
    test_steps = np.ceil(test_data_gen.samples / 32)

    model = build_and_train_model(train_data_gen, test_data_gen, train_steps, test_steps)

# このスクリプトが直接実行された場合のみmain()を実行
if __name__ == "__main__":
    main()