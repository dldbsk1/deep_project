import cv2
import sys, random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image


def cut_image(image_file, num_files, m, n, output_filename):
    # 이미지 파일 읽기
    tiles = []
    labels = []
    dataset = []

    for imgno in range(1, num_files + 1):
        image = cv2.imread(image_file + str(imgno) + '.jpg')

        # 오류 검출
        if image is None:
            print("이미지를 불러오는데 실패하였습니다.")
            return
        # 이미지 크기 조정 (글자 이미지 부분을 제외한 좌/우, 위/아래 여백 제거)
        H, W = image.shape[0], image.shape[1]
        cutW1 = int(W * 0.04761)  # 왼쪽 여백 비율
        cutW2 = int(W * 0.06666)  # 오른쪽 여백 비율
        cutH1 = int(H * 0.048821)  # 위 여백 비율
        cutH2 = int(H * 0.02693)  # 아래 여백 비율
        print("처리 이미지 파일 이름: ", image_file + str(imgno) + '.jpg')
        print("입력 이미지 형상: ", W, H)
        image = image[cutH1: H - cutH2, cutW1: W - cutW2, ]  # 전체 이미지에서 왼쪽/오른쪽, 위/아래의 여백을 자르기
        if image.shape[0] % n != 0:  # 나머지가 0이 아닐 경우 높이 - 나머지
            image = image[:-(image.shape[0] % n), :]
        if image.shape[1] % m != 0:
            image = image[:, :-(image.shape[1] % m)]

        # 이미지를 m x n 크기로 나누기
        tile_height = image.shape[0] // n
        tile_width = image.shape[1] // m
        LeftCut, RightCut = 0, 0  # 타일내에 좌/우 여백이 많을 경우 여백 제거 (픽셀 단위로 입력)
        TopCut, BottomCut = 0, 0  # 타일내에 위/아래 여백이 많을 경우 여백 제거 (픽셀 단위로 입력)
        for i in range(n):
            for j in range(m):
                # 타일의 좌측 위부터 계산
                top = i * tile_height
                left = j * tile_width
                # 타일 이미지 가져오기
                tile = image[top:top + tile_height, left:left + tile_width].copy()
                tile = tile[LeftCut:tile.shape[0] - RightCut, TopCut: tile.shape[1] - BottomCut]
                tile[0:4, 0:tile_width] = [255]  # 경계영역(위)  글자/잡음 제거
                # (경계영역의 픽셀을 255로 채움)
                tile[0:tile_height, 0:4] = [255]  # 경계영역(좌)  글자/잡음 제거
                # (경계영역의 픽셀을 255로 채움)
                tile[tile_height - 4:tile_height, 0:tile_width] = [255]  # 경계영역(아래)  글자/잡음 제거
                # (경계영역의 픽셀을 255로 채움)
                tile[0:tile_height, tile_width - 4:tile_width] = [255]  # 경계영역(위)  글자/잡음 제거
                # (경계영역의 픽셀을 255로 채움)

                tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)  # 회색조로 변경
                tile = cv2.resize(tile, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)  # 32X32로 해상도 변경
                tile = cv2.bilateralFilter(tile, -1, 10, 5)  # 이미지 선명하게
                tiles.append(tile)  # 현재 타일을 이미지 리스트에 추가
                labels.append(i % 14)  # 현태 타이블의 정답(Label)을 정답 리스트에 추가

    test_data = np.array(tiles)  # 넘파이 배열로 변환
    test_data = 255 - test_data  # 0 검은색, 255 흰색으로 변환
    labels = np.array(labels)
    print("shape of test_data", test_data.shape)
    print("shape of labels", labels.shape)
    dataset = test_data, labels

    # 2D (n, 32,32) 형태로 변환하여 저장
    with open(output_filename + '2D' + '.pkl', 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 3D (n,1, 32,32) 형태로 변환하여 저장
    test_data1 = np.expand_dims(test_data, axis=1)
    dataset = test_data1, labels
    print("3D 형상", test_data1.shape)
    with open(output_filename + '3D' + '.pkl', 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 1D (n,1024) 형태로 변환하여 저장
    test_data_1d = test_data.reshape(test_data.shape[0], 32 * 32)
    dataset_1d = test_data_1d, labels
    print("1D 형상", test_data_1d.shape)
    with open(output_filename + '1D' + '.pkl', 'wb') as f:
        pickle.dump(dataset_1d, f, protocol=pickle.HIGHEST_PROTOCOL)

    return tiles


def main():
    # 이미지 자르기 및 타일 이미지 저장
    cut_image("IMG", 12, 18, 28, "hwdata")

    # 명령어 입력 확인
    if len(sys.argv) != 6:  # 실행파일, 입력파일이름, 파일의 수, 행, 열, 출력파일이름 총 6개 입력이 되지 않을 경우 오류
        print("잘못된 입력입니다. 입력파일이름, 파일의 수, 행, 열, 출력파일이름을 이용해 주세요")
        return
    # argv[]로 각 인덱스마다 값을 확인 및 입력
    image_file_name = sys.argv[1]
    num_files = int(sys.argv[2])
    m = int(sys.argv[3])
    n = int(sys.argv[4])
    output_filename = sys.argv[5]


if __name__ == "__main__":
    main()