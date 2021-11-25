# are_you_ok

# 개발배경 및 목적

최근 집이나 보육시설에 아이들의 안전을 위한 스마트 홈카메라의 설치비중이 증가하고 있습니다. 

하지만 바쁜 현대 사회를 살면서 실시간으로 찍히는 영상을 모두 확인하는 것은 쉬운 일이 아니고, 아이들이 사고 사실을 알리지 않는 경우에 보호자가 사고 상황을 인지하는 것은 힘든 것이 현실입니다. 

그래서 스마트 홈카메라 기반 실시간 관찰 프로그램 Are you OK? 는 사고 발생시 실시간으로 이상행동을 AI로 분석해서 사고 사실을 알려주어 빠르게 사고 상황 파악을 가능하게 도와주는 것을 목표로 합니다.

# 개발환경 및 개발언어

개발 환경은 Window에서 진행되었으며, 개발 언어는 Python을 사용하였습니다.

# 시스템 구성 및 아키텍처

- 데이터 전처리: OpenCV
- 데이터 학습: Tensorflow
- Openpose-multi-person을 통해 한번에 여러 사람의 스켈레톤 인식 가능

[협업 툴]

- Github
- Discord / Slack

# 프로젝트 주요기능

‘Are you OK?’의 주요 기능은 다음과 같습니다.

1. GUI를 실행하고 파일 추가 혹은 카메라 선택 버튼을 누른 뒤, 시작 버튼을 클릭합니다.
2. 시작 버튼을 누르면, 동영상 파일 혹은 카메라에 찍히는 현장에 대해 폭력 검출을 시작합니다.
3. 실행되면서, log 값들이 GUI의 log 값 출력 프레임에 나타나 폭력 검출에 대한 값을 사용자가 볼 수 있도록 합니다.

# 기대효과 및 활용분야

현재, 가볍게 개발하여 저사양 컴퓨터로는 실시간 실행이 불가하지만, 고사양 제품을 이용하면 실시간 행동 분석 시스템 ‘Are you OK’의 실행 가능성이 높다고 기대하는 바입니다.

폭력 행동을 검출하는 알고리즘을 이용하여, TV 혹은 영화 방영 전 과도한 폭력행동을 포착할 수 있으므로 심의를 제한하는데 활용할 수 있습니다.

## Require
we checked in window 10

Python 3.7

Tensorflow 2.3

OpenCV 4.3.5

We recommend highend CPU and GPU system with CUDA and OpenCV GPU setting.

## Files
ayo.py

human_problam.txt

/models


## Models

models download in here -> https://drive.google.com/file/d/1XBi5n1zkZym_Ug8XL3NZtdZrFVfzLpSK/view?usp=sharing
