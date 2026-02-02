from ultralytics import YOLO
import os
from training_config import train_config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #OpenMP 라이브러리가 2번 링크되는 것을 허용
os.environ['CUDA_VISIBLE_DEVICES']='0' #1번 GPU를 사용하겠다.


model = YOLO('yolov8m-cls.pt') #선정된 model입력
model.to('cuda') #gpu에서 모델 학습하겠다.

#stryaml='E:/PublicDB/PETSKINDISEASE/TRANFORMDATA/TRAIN/DOG/TFRECORD/data.yaml'
stryaml = 'E:/PublicDB/PETSKINDISEASE/TRANFORMDATA/YOLO'
if __name__ == '__main__':

    results = model.train(data=stryaml,**train_config)
    #results = model.val()
    print(results)
    print(type(model.names),len(model.names))