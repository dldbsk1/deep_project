import pickle
import matplotlib.pyplot as plt


def load_data(file_path):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def plot_images_in_batches(data, batch_size=10):
    num_images = len(data)
    num_batches = (num_images // batch_size) + 1 if num_images % batch_size != 0 else (num_images // batch_size)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_images)

        batch_data = data[start_idx:end_idx]
        num_images_in_batch = len(batch_data)

        fig, axes = plt.subplots(1, num_images_in_batch, figsize=(2 * num_images_in_batch, 2))
        if num_images_in_batch == 1:
            axes = [axes]  # 하나의 이미지만 있을 경우를 대비하여 리스트로 변환

        for i in range(num_images_in_batch):
            image = batch_data[i].squeeze()  # (1, 32, 32) -> (32, 32)
            axes[i].imshow(image, cmap='gray')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()


# 예시로 피클 파일 경로를 지정하여 데이터 로드 및 출력
file_path = 'Train_3Daug.pkl'
data, _ = load_data(file_path)  # 라벨은 사용하지 않기 때문에 _ 로 표시합니다.
plot_images_in_batches(data, batch_size=10)
