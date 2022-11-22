from flask import Flask, request
import cv2
import numpy as np
import json
import typing as ty
from flask_cors import CORS
import base64

from marsemseg.preprocessors import Pipeline, PipelineFactory
from marsemseg.inference import predict_image
from marsemseg.segmentors import ISegmentation, SegmentorFactory

app = Flask(__name__)
CORS(app)

app.config["UPLOAD_FOLDER"] = "static/files"

SHAPE = (384, 512)
ckpt_file = "torch_script_best_mIoU_iter_4950.pth"
config_file = "pspnet_r50-d8_512x1024_40k_cityscapes.py"

with open("img_norm_cfg.json", "r") as f:
    img_norm_dict = json.load(f)

pipeline = PipelineFactory(
    backend="torchscript",
    mean=img_norm_dict["mean"],
    std=img_norm_dict["std"],
    shape=SHAPE,
    convert_to_rgb=True,
).create()

segmentor: ISegmentation = SegmentorFactory(
    backend="torchscript",
    checkpoint_path=ckpt_file,
    config=config_file,
    shape=SHAPE,
    device="cuda",
    half=True,
).create()

# Test API route
@app.route("/test")
def test():
    return {"test": "It works!"}


@app.route("/file-upload", methods=["POST"])
def file_upload():
    if request.files:
        print()

        file_obj = request.files["myFile"]

        file_stream = file_obj.stream
        file_stream.seek(0)

        img_array = np.array(bytearray(file_stream.read()), dtype=np.uint8)

        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        predicted = predict_image(img, segmentor, pipeline, SHAPE, show=False)
        _, result_arr = cv2.imencode('.png', predicted)
        result_bytes = result_arr.tobytes()
        b64prediction = base64.b64encode(result_bytes).decode()
        cv2.imwrite("test.png", predicted)

        return {"success": True, "mask": b64prediction}


if __name__ == "__main__":
    app.run(debug=True)
