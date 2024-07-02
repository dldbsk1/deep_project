import pickle
import numpy as np
#여기부턴 확인하는 코드를 위한 임포트
#제출할 땐 헷갈리니 지워내기
import random
import matplotlib.pyplot as plt

# 우리조 피클 파일 읽기
with open('hwdata3D.pkl', 'rb') as f:
    data1 = pickle.load(f)
xdata1_train, tdata1_train = data1
#xdata1_train = np.expand_dims(xdata1_train,axis=1)
#tdata1_train = np.expand_dims(tdata1_train,axis=1)
print("배열의 차원 수:", xdata1_train.ndim,tdata1_train.ndim )
print(type(data1))
print(data1[0].shape)


# 우리조_Aug 피클 파일 읽기
with open('hwdata3Daug.pkl', 'rb') as f:
    data2 = pickle.load(f)
xdata2_train, tdata2_train = data2
xdata2_train = np.expand_dims(xdata2_train,axis=1)
#tdata2_train = np.expand_dims(tdata2_train,axis=1)
print("배열의 차원 수:", xdata2_train.ndim,tdata2_train.ndim )
print(type(data2))
print(data2[0].shape)


# 다른조 피클 파일 읽기
with open('Train_3D.pkl', 'rb') as f:
    data3 = pickle.load(f)
xdata3_train, tdata3_train = data3
#xdata3_train = np.expand_dims(xdata3_train,axis=1)
#tdata3_train = np.expand_dims(tdata3_train,axis=1)
print("배열의 차원 수:", xdata3_train.ndim,tdata3_train.ndim )
print(type(data3))
print(data3[0].shape)


# 다른조_Aug 피클 파일 읽기
with open('Train_3Daug.pkl', 'rb') as f:
    data4 = pickle.load(f)
xdata4_train, tdata4_train = data4
xdata4_train = np.expand_dims(xdata4_train,axis=1)
#tdata4_train = np.expand_dims(tdata2_train,axis=1)
print("배열의 차원 수:", xdata4_train.ndim,tdata4_train.ndim)
print(type(data4))
print(data4[0].shape)

# 데이터 합치기
x_finalData = np.concatenate([xdata1_train, xdata2_train, xdata3_train, xdata4_train])
t_finalData = np.concatenate([tdata1_train, tdata2_train, tdata3_train, tdata4_train])

# 테스트 데이터 구성
test_data = np.array(x_finalData)  # 넘파이 배열로 변환
test_data = 255 - test_data  # 0 검은색, 255 흰색으로 변환
labels = t_finalData  # 라벨 합치기

# 합친 데이터를 새로운 피클 파일에 저장하기
finalData = x_finalData, t_finalData
print("형상", x_finalData.shape)
with open('finalData4D.pkl', 'wb') as f:
    pickle.dump(finalData, f, protocol=pickle.HIGHEST_PROTOCOL)

#잘 됐나 확인
def load_data(file_path):
    with open(file_path, 'rb') as f:
        checkset = pickle.load(f)
    return checkset
def plot_random_images(data, labels, num_images):
    fig, axes = plt.subplots(5, 10, figsize=(15, 8))
    indexes = np.random.choice(len(data), size=num_images, replace=False)

    for i, ax in enumerate(axes.flat):
        idx = indexes[i]
        image = data[idx].squeeze()  # (1, 32, 32) -> (32, 32)
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Label: {labels[idx]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

check_data, check_labels = load_data('finalData4D.pkl')
plot_random_images(check_data, check_labels, 50)

print(len(data1))  # 데이터1의 길이 확인
print(len(data2))  # 데이터2의 길이 확인
print(len(data3))  # 데이터3의 길이 확인
print(len(data4))  # 데이터3의 길이 확인
print(type(data1[0]))  # 데이터1의 첫 번째 요소의 자료형 확인
print(type(data2[0]))  # 데이터2의 첫 번째 요소의 자료형 확인
print(type(data3[0]))  # 데이터3의 첫 번째 요소의 자료형 확인
print(type(data4[0]))  # 데이터3의 첫 번째 요소의 자료형 확인
