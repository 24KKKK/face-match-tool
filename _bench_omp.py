import os, sys, time
import warnings, logging
warnings.filterwarnings('ignore'); logging.disable(logging.WARNING); os.environ['ORT_LOGGING_LEVEL']='3'
import numpy as np
from insightface.app import FaceAnalysis
from core.io_utils import read_image, iter_media_files
omp = os.environ.get('OMP_NUM_THREADS','(默认)')
files=[p for p,k in iter_media_files(r'D:/图片分类/待整理相册', exclude=['_thumb']) if k=='image'][:10]
app=FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640,640))
for p in files[:2]:
    img,_=read_image(p)
    if img is not None: app.get(img)
t0=time.time(); n=0
for p in files:
    img,_=read_image(p)
    if img is None: continue
    n+=len(app.get(img))
t=time.time()-t0
print(f'OMP_NUM_THREADS={omp:<8} 推理 {len(files)} 张 {t:5.2f}s ({t/len(files)*1000:.0f} ms/张) 检出 {n}')
