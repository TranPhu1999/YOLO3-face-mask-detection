# YOLO_FACE_MASK_DETECTION
In this project, I:
- Train a Yolov3 model to detect Face mask wearing (Tensorflow, OpenCV, Numpy)
- Build a Flask API that receive image and send image result to a web interface (flask)
- Build a web interface that Upload image from user then send it to Flask API, receive and display the output (HTML, CSS, JavaSript)
|![Input](https://github.com/TranPhu1999/TranDucPhu_porfolio/blob/main/images/download.png) | ![Output](https://github.com/TranPhu1999/TranDucPhu_porfolio/blob/main/images/maksssksksss0.png)|
Face wearing mask detection
- Training data: https://drive.google.com/drive/folders/1uokTkVbIpzHqevw3_211jj6iwUmr1XkS?usp=sharing
- Weights: https://drive.google.com/file/d/1gLYLXFgY4j2qpC10NqUdBaP72i12NnBS/view?usp=sharing
# Set up enviroment: 
`conda env create -f conda-cpu.yml`
`conda activate yolov3-cpu`
# Run detector:
`python app.py`
# Reference:
- Yolo: https://pjreddie.com/YoloV3 model detect Face wearing darknet/yolo/
- Flask: https://flask.palletsprojects.com/en/1.1.x/quickstart/?fbclid=IwAR1Kgoodj-z5PDX_YSMSB2kkm3uX6LjU6rZsUNDdkLAV_-nvuq1-vryL_Jo
- Train Yolo: https://miai.vn/2019/08/09/yolo-series-2-cach-train-yolo-de-detect-cac-object-dac-thu/
 https://www.youtube.com/watch?v=_UqmgHKdntU&t=674s
- Front-end: https://speckyboy.com/custom-file-upload-fields/?fbclid=IwAR1m_7rOddeyapKW1eHkA7XZuZVsL5fmwAhR7uESn8taYHxdpDtk0T8pbVQ

