import time
from PIL import Image
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from flask import Flask, flash, render_template, request, Response, jsonify, send_from_directory, abort, redirect,url_for
import os

#Disable AVX instruction
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#PRETRAINED
print('loading pretrained model...')
classes_path = './data/labels/coco.names'
weights_path = './weights/yolov3.tf'
size = 416                      # size images are resized to for model
output_path = './detections/'   # path to output folder where images with detections are saved
num_classes = 80                # number of classes in model
yolo = YoloV3(classes=num_classes)    
yolo.load_weights(weights_path).expect_partial()
print('weights loaded')
class_names = [c.strip() for c in open(classes_path).readlines()]
print('classes loaded')


#CUSTOM
print('loading custom model...')
classes_path = './data/labels/yolo_mask.names'
weights_path = './weights/yolov3_mask.tf'
size = 416                      # size images are resized to for model
output_path = './detections/'   # path to output folder where images with detections are saved
num_classes = 3                # number of classes in model
yoloCustom = YoloV3(classes=num_classes)
yoloCustom.load_weights(weights_path).expect_partial()
print('weights loaded')
class_names_custom = [c.strip() for c in open(classes_path).readlines()]
print('classes loaded')



UPLOAD_FOLDER = "Upload"

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# API that return the main UI
@app.route('/')
def main_UI():   
    return render_template('uploadpage.html')

@app.route('/Upload', methods = ['GET','POST'])
def uploadFile():
    if request.method== 'POST':
        imgsrcstring = request.form['data']
        if imgsrcstring:
            imgdata = imgsrcstring.split(',')[1]
            decoded = base64.b64decode(imgdata)
            img = Image.open(BytesIO(decoded))          
            filename = secure_filename(request.form['name'])
            img.save(os.path.join(app.config['UPLOAD_FOLDER'],request.form['name']))
            if(request.form['model'] == 'pretrained'):
                img = return_image(img,filename,False)
            else:
                img = return_image(img,filename,True)
            img = Image.open(output_path + 'detection_' + filename)
            ratio = min(900/img.size[0],900/img.size[1])
            img = img.resize((int(img.size[0]*ratio),int(img.size[1]*ratio)),Image.ANTIALIAS)
            
            rawBytes = BytesIO()
            img.save(rawBytes,"JPEG")
            rawBytes.seek(0)
            img_base64 = base64.b64encode(rawBytes.getvalue()).decode('ascii')
            uri= "data:%s;base64,%s"%("image/jpeg",img_base64)
            image=uri
            return jsonify({'detected':image})
        else:
            return redirect(request.url)
 
#returns image with detections of pretrained model
def return_image(image,filename,custom=False):    

    img_raw = tf.image.decode_image(
        open(os.path.join(app.config['UPLOAD_FOLDER'],filename), 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)
    t1 = time.time()
    if(custom == True):
        boxes, scores, classes, nums = yoloCustom(img)
    else:
        boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    print('detections:')
    
    if(custom == True):
        class_names_local = class_names_custom
    else:
        class_names_local = class_names
        
    for i in range(nums[0]):
        print('\t{}, {}, {}'.format(class_names_local[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names_local)
    cv2.imwrite(output_path + 'detection_' + filename, img)
    print('output saved to: {}'.format(output_path + 'detection_' + filename))
    try:
        return img
    except FileNotFoundError:
        abort(404)

def getIP():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == '__main__':
    import sys  
    app.run(debug=True, host = getIP(),port=5000, use_reloader=False)
##    Cau lenh de chay tren linux
##    app.run(debug=True, host = sys.argv[1] ,port=5000, use_reloader=False)  
