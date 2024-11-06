import os
import subprocess
import argparse

from exception.exception import InvalidPathError, InvalidExtensionError, InvalidFPSError


def extract_frames(video_path, output_path, video_name, fps):
    # 결과 폴더 생성
    video_output_dir = os.path.join(output_path, video_name)
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)

    # -y: 파일 덮어 씌우기 옵션
    command = [
        'ffmpeg', '-y', '-i', video_path, '-vf', f'fps=1/{fps}',
        f'{output_path}/{video_name}/frame_%04d.jpg'
    ]

    try:
        subprocess.run(command, text=True, capture_output=True, check=True)
        print(f"[완료] {video_path}에서 이미지 추출 완료.")

    except subprocess.CalledProcessError as e:
        print(f"[오류] {video_path} 처리 중 문제 발생: {e.stderr}")


def extract_image_from_video(args):
    input_file = os.path.abspath(args.input_file)  # 절대 경로로 변환
    output_dir = os.path.abspath(args.output_dir)  # 절대 경로로 변환
    fps = args.fps

    # 결과 폴더 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # input_file이 존재하는지 확인
    if not os.path.exists(input_file):
        raise InvalidPathError(f"'{input_file}' : 입력 파일이 없거나, 경로가 올바르지 않습니다.")
    # 확장자가 동영상인지 확인
    if not input_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise InvalidExtensionError(f"'{input_file}' : 올바른 확장자가 아닙니다.")
    # fps가 int인지 확인
    if not isinstance(fps, int):
        raise InvalidFPSError(f"'{fps}' : FPS 값이 올바르지 않습니다. 정수를 입력하세요.")

    video_name = os.path.splitext(os.path.basename(input_file))[0]  # 파일 이름 추출
    extract_frames(input_file, output_dir, video_name, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="./data/input.mp4",
                        help="이미지를 추출할 동영상의 경로 (기본값: ./data/input.mp4)")
    parser.add_argument("--output_dir", type=str, default="./data/output",
                        help="추출한 이미지를 저장할 디렉토리 경로 (기본값: ./data/output)")
    parser.add_argument("--fps", type=int, default=1,
                        help="영상에서 이미지를 추출할 초 단위 (기본값: 1)")
    args = parser.parse_args()

    extract_image_from_video(args)

