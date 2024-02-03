import os
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input
import numpy as np
from custom_layer import CustomLayer

# モデルのロード
model_path = 'resnet152_model.h5'
model = load_model(model_path, custom_objects={'CustomLayer': CustomLayer})

# クラス名の定義
class_names = ["paru", "poke"]

def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(model, img_path):
    img = preprocess_image(img_path, target_size=(224, 224))
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions) * 100
    return predicted_class_name, confidence

def main():
    folder_path = 'path_to_test_data\pal'  # 検索パス
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    total_images = len(image_files)
    print(f"Total images in the folder: {total_images}")

    class_count = {classname: 0 for classname in class_names}
    classified_images = {classname: [] for classname in class_names}  # 分類された画像のファイル名を保存するための辞書

    for i, image_file in enumerate(image_files, 1):
        img_path = os.path.join(folder_path, image_file)
        predicted_class, _ = predict_image(model, img_path)
        class_count[predicted_class] += 1
        classified_images[predicted_class].append(image_file)  # 予測されたクラスに画像ファイル名を追加
        print(f"Processing image {i}/{total_images}: {image_file}")

    for classname, files in classified_images.items():
        print(f"Class '{classname}': {len(files)} images")
        for file in files:
            print(f" - {file}")

if __name__ == "__main__":
    main()
