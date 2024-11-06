import torch
import torch.nn as nn
import os

# Import the custom layers from the models/layer directory
from models.layer.stem import Stem
from models.layer.inception_resnet_block import InceptionResNetA, InceptionResNetB, InceptionResNetC
from models.layer.reduction_block import ReductionA, ReductionB
from models.layer.scaling_layer import ScalingLayer
from models.layer.final_block import FinalBlock
from models.layer.basic_layers import Conv2dBnRelu

class InceptionResNetV2(nn.Module):
    def __init__(self, num_classes=1000, dropout_rate=0.2, scale=True):
        super(InceptionResNetV2, self).__init__()
        # 초기 레이어 정의 (Stem)
        self.stem = Stem()

        # Inception-ResNet 블록 정의
        self.inception_resnet_a = nn.Sequential(
            *[InceptionResNetA(scale=scale) for _ in range(5)]
        )
        self.reduction_a = ReductionA()

        self.inception_resnet_b = nn.Sequential(
            *[InceptionResNetB(scale=scale) for _ in range(10)]
        )
        self.reduction_b = ReductionB()

        self.inception_resnet_c = nn.Sequential(
            *[InceptionResNetC(scale=scale) for _ in range(5)]
        )

        # 최종 레이어 정의
        self.final_block = FinalBlock(num_classes=num_classes, dropout_rate=dropout_rate)

    def forward(self, x):
        # Stem을 통과
        x = self.stem(x)

        # Inception-ResNet 블록을 통과
        x = self.inception_resnet_a(x)
        x = self.reduction_a(x)
        x = self.inception_resnet_b(x)
        x = self.reduction_b(x)
        x = self.inception_resnet_c(x)

        # 최종 레이어를 통과
        x = self.final_block(x)
        return x

def inception_resnet_v2(num_classes=1000, dropout_rate=0.2, pretrained=False, scale=True, weights_path=None):
    """
    Inception-ResNet-v2 모델을 생성하는 팩토리 함수.
    Args:
        num_classes (int): 출력 클래스의 수.
        dropout_rate (float): 드롭아웃 비율.
        pretrained (bool): True일 경우, 사전 학습된 가중치를 로드.
        scale (bool): 잔차 연결에 스케일링을 적용할지 여부.
        weights_path (str): 사전 학습된 가중치 파일의 경로.
    """
    model = InceptionResNetV2(num_classes=num_classes, dropout_rate=dropout_rate, scale=scale)
    if pretrained:
        if weights_path and os.path.exists(weights_path):
            # 경로가 유효한 경우 사전 학습된 가중치를 로드
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            print(f"{weights_path}에서 사전 학습된 가중치를 로드했습니다.")
        else:
            raise ValueError("사전 학습된 가중치 경로가 유효하지 않거나 지정되지 않았습니다.")
    return model
