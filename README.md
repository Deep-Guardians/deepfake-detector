# Deepfake Detector
이 프로젝트는 멀티모달 방식으로 딥페이크를 탐지하는 모델을 구현합니다. Inception-ResNet-v2 모델을 기반으로 눈, 코, 입과 같은 얼굴 부위의 특징을 분석하여 딥페이크를 분류합니다. 만약 특정 얼굴 부위가 감지되지 않으면 전체 얼굴을 이용하여 탐지를 수행합니다.

## 프로젝트 구조
```
deepfake-detector/
├── data/
│   ├── REAL/                    # 실제 이미지 데이터 폴더
│   └── FAKE/                    # 가짜 이미지 데이터 폴더
├── models/
│   ├── inception_resnet_v2.py   # Inception-ResNet-v2 모델 정의 파일
│   ├── checkpoints/             # 모델 가중치가 저장될 폴더
│   └── layer/                   # 레이어 관련 파일들
│       ├── stem.py              # Stem 레이어 정의
│       ├── inception_resnet_block.py # Inception-ResNet 블록 정의 (A, B, C)
│       ├── reduction_block.py   # Reduction 블록 정의 (A, B)
│       ├── scaling_layer.py     # Scaling 레이어 정의
│       ├── final_block.py       # 최종 레이어 정의
│       └── basic_layers.py      # 기본 레이어 정의 (Conv2d, BatchNorm 등)
├── train_multimodal.py          # 멀티모달 학습 스크립트
├── requirements.txt             # 필요한 파이썬 패키지 목록
└── README.md                    # 프로젝트 설명 파일
```

## 설치 방법
1. 이 저장소를 클론합니다.
```
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector
```
2. 필요한 패키지를 설치합니다.
```
pip install -r requirements.txt
```
3. dlib의 얼굴 랜드마크 모델을 다운로드하고 models/ 폴더에 위치시킵니다.
* shape_predictor_68_face_landmarks.dat 파일을 [여기](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)에서 다운로드하고 압축을 풉니다.

## 사용 방법
1. 데이터 준비
* data/REAL 폴더에는 실제 이미지 데이터를,
* data/FAKE 폴더에는 가짜 이미지 데이터를 넣습니다.

2.학습 실행
* 다음 명령어를 사용하여 모델을 학습시킬 수 있습니다.
```
    python train_multimodal.py --epochs 20 --batch_size 8 --learning_rate 0.001 --real_dir ./data/REAL --fake_dir ./data/FAKE
```

## 주요 파일 설명
* train_multimodal.py: 멀티모달 방식으로 딥페이크 탐지 모델을 학습시키기 위한 메인 스크립트입니다.
* models/inception_resnet_v2.py: Inception-ResNet-v2 모델의 구조를 정의한 파일입니다.
* models/layer/: Inception-ResNet-v2 모델의 각 레이어와 블록을 정의한 파일들이 모여 있습니다.
* data/: 실제 이미지와 가짜 이미지 데이터를 저장하는 폴더입니다.
* requirements.txt: 필요한 파이썬 패키지 목록이 포함된 파일입니다.

## 기여 방법
1. 이 저장소를 포크합니다.
2. 새로운 브랜치를 생성합니다.
   ```
   git checkout -b feature/your-feature-name
   ```
3. 변경 사항을 커밋합니다.
   ```
   git commit -m "Add some feature"
   ```
4. 브랜치에 푸시합니다.
   ```
   git push origin feature/your-feature-name
   ```
5. 풀 리퀘스트를 생성합니다.

## 라이센스
이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 참조
* [dlib 라이브러리](http://dlib.net/)
* [PyTorch](https://pytorch.org/docs/stable/index.html)
* [WandB](https://wandb.ai/)