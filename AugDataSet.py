import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import cv2


def load_data(file_path):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def augment_image(image):
    augmented_images = [image]

    # 회전 (45도씩)
    for angle in range(45, 360, 45):
        rotated_image = rotate(image, angle, reshape=False)
        augmented_images.append(rotated_image)

    # 이미지 축소
    scaled_image = cv2.resize(image, (16, 16), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros_like(image, dtype=np.uint8)
    y_offset = (image.shape[0] - 16) // 2
    x_offset = (image.shape[1] - 16) // 2
    canvas[y_offset:y_offset + 16, x_offset:x_offset + 16] = scaled_image
    augmented_images.append(canvas)

    return np.array(augmented_images)

def augment_data(data, labels):
    augmented_data = []
    augmented_labels = []

    for img, lbl in zip(data, labels):
        augmented_imgs = augment_image(img)
        augmented_data.append(augmented_imgs)
        augmented_labels.extend([lbl] * len(augmented_imgs))

    augmented_data = np.concatenate(augmented_data, axis=0)
    augmented_labels = np.array(augmented_labels)

    return augmented_data, augmented_labels


def plot_random_images(data, labels, num_images):
    random_indices = random.sample(range(len(data)), num_images)
    fig, axes = plt.subplots(5, 10, figsize=(15, 7))
    axes = axes.flatten()

    for ax, idx in zip(axes, random_indices):
        ax.imshow(data[idx], cmap='gray')
        ax.set_title(f'Label: {labels[idx]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def save_data_to_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved augmented data to {file_path}")
def main():
    file_path = 'hwdata2D.pkl'
    # 데이터 로드
    data, labels = load_data(file_path)
    # 데이터 증강
    augmented_data, augmented_labels = augment_data(data, labels)
    # 증강된 데이터를 넘파이 배열로 저장
    save_data_to_pickle((augmented_data, augmented_labels), 'hwdata3Daug.pkl')
    # 랜덤한 이미지 50개 출력
    plot_random_images(augmented_data, augmented_labels, 100)

if __name__ == "__main__":
    main()
