import cv2
import tensorflow as tf

from tensorflow.keras.layers import Input, UpSampling2D, Add, Concatenate, Lambda
from keras_resnet.models import ResNet18
from .layers import ApproximateBinaryMap, ConvBnRelu, DeConvMap
from utils.processor import PostProcessor
from utils.bbox import order_boxes4nom

class DBNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = self._build_model()
        self.post_processor = PostProcessor(min_box_score=0.5, max_candidates=1000)
        

    def _build_model(self, k=50):
        image_input = Input(shape=(None, None, 3), name='image')
        backbone = ResNet18(inputs=image_input, include_top=False)
        
        C2, C3, C4, C5 = backbone.outputs
        in2 = ConvBnRelu(256, 1, name='in2')(C2)
        in3 = ConvBnRelu(256, 1, name='in3')(C3)
        in4 = ConvBnRelu(256, 1, name='in4')(C4)
        in5 = ConvBnRelu(256, 1, name='in5')(C5)
        
        # The pyramid features are up-sampled to the same scale and cascaded to produce feature F
        upsampled_in5 = UpSampling2D(size=(2, 2), name='up5')(in5)
        out4 = Add(name='add_out4')([upsampled_in5, in4])  # Replace direct addition with Add layer

        upsampled_out4 = UpSampling2D(size=(2, 2), name='up4')(out4)
        out3 = Add(name='add_out3')([upsampled_out4, in3])

        upsampled_out3 = UpSampling2D(size=(2, 2), name='up3')(out3)
        out2 = Add(name='add_out2')([upsampled_out3, in2])
        
        P5 = tf.keras.Sequential([ConvBnRelu(64, 3), UpSampling2D(8)], name='P5')(in5)
        P4 = tf.keras.Sequential([ConvBnRelu(64, 3), UpSampling2D(4)], name='P4')(out4)
        P3 = tf.keras.Sequential([ConvBnRelu(64, 3), UpSampling2D(2)], name='P3')(out3)
        P2 = ConvBnRelu(64, 3, name='P2')(out2)
        
        # Calculate DBNet maps
        fuse = Concatenate(name='fuse')([P2, P3, P4, P5]) # (batch_size, /4, /4, 256)
        binarize_map = DeConvMap(64, name='probability_map')(fuse)
        threshold_map = DeConvMap(64, name='threshold_map')(fuse)
        approximate_binary_map_layer = ApproximateBinaryMap(k=k, name='approximate_binary_map')
        thresh_binary = approximate_binary_map_layer([binarize_map, threshold_map])
        
        return tf.keras.Model(
            inputs = image_input, 
            outputs = [binarize_map, threshold_map, thresh_binary], 
            name = 'DBNet'
        )


    def resize_image_short_side(self, image, image_short_side=736):
        height, width, _ = image.shape
        if height < width:
            new_height = image_short_side
            new_width = int(round(new_height / height * width / 32) * 32)
        else:
            new_width = image_short_side
            new_height = int(round(new_width / width * height / 32) * 32)
        return cv2.resize(image, (new_width, new_height))

    def predict_one_page(_self, raw_image):
        image = _self.resize_image_short_side(raw_image).astype(float) / 255.0
        binarize_map = _self.model(tf.expand_dims(image, 0), training=False)[0]     
        batch_boxes, batch_scores = _self.post_processor(binarize_map.numpy(), [raw_image.shape[:2]])
        boxes = order_boxes4nom(batch_boxes[0])
        return boxes