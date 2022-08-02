import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def IOU_metric( y_true, y_pred ):
    # iou as metric for bounding box regression
    # input must be as [x1, y1, x2, y2]
    
    # AOG = Area of Groundtruth box
    true = tf.transpose(y_true)
    pred = tf.transpose(y_pred)

    pred_2 = tf.maximum( pred[2], pred[0] )
    pred_3 = tf.maximum( pred[3], pred[1] )
    AoG = tf.abs(true[2] - true[0]) * tf.abs(true[3] - true[1])
    
    # AOP = Area of Predicted box
    AoP = tf.abs(pred_2 - pred[0]) * tf.abs(pred_3 - pred[1])

    # overlaps are the co-ordinates of intersection box
    overlap_0 = tf.maximum(true[0], pred[0])
    overlap_1 = tf.maximum(true[1], pred[1])
    overlap_2 = tf.minimum(true[2], pred_2)
    overlap_3 = tf.minimum(true[3], pred_3)

    overlap_22 = tf.maximum( overlap_2, overlap_0 )
    overlap_32 = tf.maximum( overlap_3, overlap_1 )

    # intersection area
    intersection = (overlap_22 - overlap_0) * (overlap_32 - overlap_1)

    # area of union of both boxes
    union = AoG + AoP - intersection
    
    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = tf.clip_by_value(iou, 0.0, 1.0)

    return iou