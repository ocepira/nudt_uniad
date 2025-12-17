FROM python:3.8

WORKDIR /project

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgl1 libglib2.0-0 && \
#     rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.7 --index-url https://download.pytorch.org/whl/cu111 \
    && pip install --no-cache-dir -r requirements.txt

RUN pip install opencv-python 

    
COPY . ./

# 自动驾驶测试
CMD ["python", "test.py"]

# 自动驾驶攻击
# CMD ["python", "attack.py"]

# 自动驾驶防御
# CMD ["python", "defense.py"]