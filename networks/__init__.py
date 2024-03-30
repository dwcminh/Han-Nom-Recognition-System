from .crnn import CRNN
from .dbnet import DBNet

import onnxruntime

def load_models(use_onnx = False):
    if not use_onnx:
        det_model = DBNet()
        rec_model = CRNN()
        det_model.model.load_weights('./weights/DBNet.h5')
        rec_model.model.load_weights('./weights/CRNN.h5')
    else:
        det_model = onnxruntime.InferenceSession('./weights/DBNet.onnx')
        rec_model = onnxruntime.InferenceSession('./weights/CRNN.onnx')

    return det_model, rec_model