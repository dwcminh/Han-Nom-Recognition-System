import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from networks.crnn import CRNN
from networks.dbnet import DBNet
import tensorflowjs as tfjs


if __name__ == '__main__':
    dbnModel = DBNet()
    crnnModel = CRNN()

    dbn = dbnModel._build_model()
    dbn.load_weights("./weights/DBNet.h5")
    tf.saved_model.save(dbn, "weights/DBNet")
    
    crnn = crnnModel._build_model()
    crnn.load_weights("./weights/CRNN.h5")
    tf.saved_model.save(dbn, "weights/CRNN")
        
    tfjs.converters.convert_tf_saved_model("weights/DBNet", "web/assets/weights/DBNet")
    tfjs.converters.convert_tf_saved_model( "weights/CRNN", "web/assets/weights/CRNN")
    
    # ov.save_model(DBNet, './weights/compressed/DBNet.xml')
    # ov.save_model(CRNN, './weights/compressed/CRNN.xml')