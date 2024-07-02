import psutil
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pickle
from PIL import Image
from torch.utils.data import DataLoader
from resNet18_2 import ResNet18, test
import time  # Import the time module

# 테스트 데이터셋 로드
with open('DATA1_3D.pkl', 'rb') as f:
    dataset = pickle.load(f)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def print_parameter_count(model):
    print("Model Parameter Counts by Layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()} parameters")

# 테스트를 위해 데이터셋 변환
testset = [(transform(Image.fromarray(image.squeeze())), label) for image, label in zip(dataset[0], dataset[1])]
testloader = DataLoader(testset, batch_size=10, shuffle=False)

# 모델 초기화 및 사전 훈련된 가중치 로드
model = ResNet18()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
model.load_state_dict(torch.load('resnet18_weights2.pkl', map_location=device))
# 평가를 위한 손실 함수
criterion = nn.CrossEntropyLoss()

# 테스트에 걸린 시간 측정 시작
start_time = time.time()

# 테스트 데이터셋에서 모델 평가
test_loss, test_acc = test(model, criterion, testloader, device)

# 테스트에 걸린 시간 측정 종료
end_time = time.time()
elapsed_time = end_time - start_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 테스트 데이터셋에서 모델 평가
test_loss, test_acc = test(model, criterion, testloader, device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
# print_parameter_count(model)

print(f'Test Time: {elapsed_time:.2f} seconds')
print(count_parameters(model))

# GPU 메모리 할당량 확인
print(torch.cuda.memory_allocated())
print(torch.cuda.max_memory_allocated())

# CPU 메모리 사용량 확인
cpu_memory_usage = psutil.virtual_memory().percent
print("CPU 메모리 사용량: {}%".format(cpu_memory_usage))
