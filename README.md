
# SOCAR-AI-HACKATHON  
> 3팀: 차량 운전자 졸음 및 부주의 탐지  
  
## Environment
 - Python 3.8
 - Anaconda Jupyter Notebook
 - OpenCV
 - Torch

## Install
```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# torch install
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
# start
jupyter notebook
```

## Files
- classification_model/* : 성능이 잘나온 pth 파일
- demo.ipynb: CV를 통한 detection demo 
- EDA_Result.ipynb: 이미지 및 학습 결과 시각화
- final.ipynb: 모델 학습 

### images
- 아래 링크 이미지를 사용하였고, github에 따로 업로드 하지 않았습니다.
- https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=173
