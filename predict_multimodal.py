import os
import argparse
import torch
import cv2
import timm
from torchvision import transforms
from exception.exception import InvalidPathError, InvalidExtensionError
from utils.face_utils import extract_facial_regions

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((299, 299)),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 모델 로드 함수
def load_multimodal_models(weights_path, device):
    # timm에서 inception_resnet_v2 모델 불러오기 (기본 num_classes=1000으로 불러온 후 classif 수정)
    model_eye = timm.create_model('inception_resnet_v2')
    model_nose = timm.create_model('inception_resnet_v2')
    model_mouth = timm.create_model('inception_resnet_v2')

    # classif를 Sequential 구조로 덮어씌우기
    for model in [model_eye, model_nose, model_mouth]:
        model.classif = torch.nn.Sequential(
            torch.nn.Linear(model.classif.in_features, 128),
            torch.nn.ReLU()
        )

    final_layer = torch.nn.Sequential(
        torch.nn.Linear(128 * 3, 1),
        torch.nn.Sigmoid()
    )

    # 모델 가중치 불러오기
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model_eye.load_state_dict(checkpoint['model_eye_state_dict'])
    model_nose.load_state_dict(checkpoint['model_nose_state_dict'])
    model_mouth.load_state_dict(checkpoint['model_mouth_state_dict'])
    final_layer.load_state_dict(checkpoint['final_layer_state_dict'])

    model_eye.eval().to(device)
    model_nose.eval().to(device)
    model_mouth.eval().to(device)
    final_layer.eval().to(device)

    return model_eye, model_nose, model_mouth, final_layer


# 예측 함수
def predict(input_path, model_eye, model_nose, model_mouth, final_layer, device):
    if not os.path.exists(input_path):
        raise InvalidPathError(f"'{input_path}' : 경로가 올바르지 않습니다.")

    if os.path.isfile(input_path):
        if not input_path.endswith(('.jpg', '.jpeg', '.png')):
            raise InvalidExtensionError(f"'{input_path}' : 적절한 파일 확장자가 아닙니다.")
        return process_image(input_path, model_eye, model_nose, model_mouth, final_layer, device)
    elif os.path.isdir(input_path):
        predictions = []
        for frame_file in sorted(os.listdir(input_path)):
            if not frame_file.endswith(('.jpg', '.jpeg', '.png')):
                raise InvalidExtensionError(f"'{frame_file}' : 적절한 파일 확장자가 아닙니다.")
            frame_path = os.path.join(input_path, frame_file)
            fake_prob, real_prob = process_image(frame_path, model_eye, model_nose, model_mouth,
                                                 final_layer, device)
            predictions.append((fake_prob, real_prob))

        real_prob = sum(pred[1] for pred in predictions) / len(predictions)
        fake_prob = 1 - real_prob
        return real_prob, fake_prob

def process_image(image_path, model_eye, model_nose, model_mouth, final_layer, device):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("이미지를 로드할 수 없습니다.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    eye, nose, mouth, fallback = extract_facial_regions(image_rgb)
    if eye is None or nose is None or mouth is None:
        if fallback is not None:
            eye, nose, mouth = fallback, fallback, fallback
        else:
            raise ValueError("얼굴 부위를 감지할 수 없습니다.")

    eye = transform(eye).unsqueeze(0).to(device)
    nose = transform(nose).unsqueeze(0).to(device)
    mouth = transform(mouth).unsqueeze(0).to(device)

    eye_features = model_eye(eye)
    nose_features = model_nose(nose)
    mouth_features = model_mouth(mouth)

    combined_features = torch.cat((eye_features, nose_features, mouth_features), dim=1)
    output = final_layer(combined_features)

    real_prob = output.item()
    fake_prob = 1 - real_prob
    return fake_prob, real_prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="딥페이크 탐지 모델 예측 스크립트")
    parser.add_argument("--input_path", type=str, required=True, help="예측할 파일의 경로(ex. sample.jpg, data/input)")
    parser.add_argument("--weights_path", type=str, required=True, help="모델 가중치가 저장된 디렉토리 경로")
    parser.add_argument("--gpu", type=int, default=0, help="사용할 GPU ID (기본값: 0)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu < torch.cuda.device_count() else "cpu")
    print(f"Using device: {device}")

    model_eye, model_nose, model_mouth, final_layer = load_multimodal_models(args.weights_path, device)
    fake_prob, real_prob = predict(args.input_path, model_eye, model_nose, model_mouth, final_layer, device)
    print(f"딥페이크 확률: {fake_prob * 100:.2f}%, 실제 확률: {real_prob * 100:.2f}%")
