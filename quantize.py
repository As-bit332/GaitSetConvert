import torch
from onnxruntime.quantization.calibrate import CalibrationDataReader
from torch.utils.data import DataLoader, Dataset
import onnx
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod, quantize_static

import vai_q_onnx
from datetime import datetime
import numpy as np
import argparse

from model.initialization import initialization
from model.utils import evaluation
from config import conf
def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 80000')
    parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
    parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
    opt = parser.parse_args()
    #m = initialization(conf, test=opt.cache)[0]

    # load model checkpoint of iteration opt.iter
    #print('Loading the model of iteration %d...' % opt.iter)
    #m.load(opt.iter)
    #print('Transforming...')
    #time = datetime.now()
    #dr = m.transform('test', opt.batch_size)
        # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = "work/ryzenai_gaitset_preprocessed.onnx"

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = "work/ryzenai_gaitset_quantized.onnx"

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    calibration_dataset_path = "data/"

    # `dr` (Data Reader) is an instance of ResNetDataReader, which is a utility class that 
    # reads the calibration dataset and prepares it for the quantization process.
    #dr = resnet_calibration_reader(calibration_dataset_path)

    # `quantize_static` is a function that applies static quantization to the model.
    # The parameters of this function are:
    # - `input_model_path`: the path to the original, unquantized model.
    # - `output_model_path`: the path where the quantized model will be saved.
    # - `dr`: an instance of a data reader utility, which provides data for model calibration.
    # - `quant_format`: the format of quantization operators. Need to set to QDQ or QOperator.
    # - `activation_type`: the data type of activation tensors after quantization. In this case, it's QUInt8 (Quantized Int 8).
    # - `weight_type`: the data type of weight tensors after quantization. In this case, it's QInt8 (Quantized Int 8).
    # - `enable_dpu`: (Boolean) determines whether to generate a quantized model that is suitable for the DPU. If set to True, the quantization process will create a model that is optimized for DPU computations.
    # - `extra_options`: (Dict or None) Dictionary of additional options that can be passed to the quantization process. In this example, ``ActivationSymmetric`` is set to True i.e., calibration data for activations is symmetrized. 
    vai_q_onnx.quantize_static(
        input_model_path,
        output_model_path,
        None,
        quant_format=vai_q_onnx.QuantFormat.QDQ,
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
        activation_type=vai_q_onnx.QuantType.QUInt8,
        weight_type=vai_q_onnx.QuantType.QInt8,
        enable_dpu=True, 
        extra_options={'ActivationSymmetric': True} 
    )
    print('Calibrated and quantized model saved at:', output_model_path)
