#!/bin/python3

import argparse
import numpy as np
import cv2
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
from pathlib import Path


quantized_model_path = r'work/ryzenai_gaitset_quantized.onnx'
model = onnx.load(quantized_model_path)


parser = argparse.ArgumentParser()
parser.add_argument('--ep', type=str, default ='cpu',choices = ['cpu','ipu'], help='EP backend selection')
opt = parser.parse_args()


providers = ['CPUExecutionProvider']
provider_options = [{}]


if opt.ep == 'ipu':
   providers = ['VitisAIExecutionProvider']
   cache_dir = Path(__file__).parent.resolve()
   provider_options = [{
                'config_file': 'vaip_config.json',
                'cacheDir': str(cache_dir),
                'cacheKey': 'modelcachekey'
            }]

session = ort.InferenceSession(model.SerializeToString(), providers=providers,
                               provider_options=provider_options)


def unpickle(file):
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

for i in range(5485):
    input_data = np.fromfile('C:/AIDEV/RyzenAI-SW-main/RyzenAI-SW-main/tutorial/GaitSet/work/input_data/' + f'{i:0>4d}.bin', dtype =np.float32)
    input_data = input_data.reshape([1, 100, 64, 44]).astype(np.float32)
    print(i)
    print(input_data.shape)
# Run the model
#print(input_data)
#print(input_data.shape)
    outputs = session.run(None, {'input': input_data})
    print(outputs[0].shape)
    outputs[0].tofile('C:/AIDEV/RyzenAI-SW-main/RyzenAI-SW-main/tutorial/GaitSet/work/output_data/' + f'{i:0>4d}.bin')
#print(outputs)
#print(outputs[0].shape)
