# import cv2
# import numpy as np
from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
import base64

from google.cloud import vision
from google.oauth2 import service_account

app = Flask(__name__)
CORS(app)

# 身元証明書のjson読み込み
credentials = service_account.Credentials.from_service_account_file('./ocr-test-357422-c9b58de1a0e2.json')
client = vision.ImageAnnotatorClient(credentials=credentials)


@app.route("/", methods=['GET', 'POST'])
def ocr():
    if request.is_json:
        data = request.get_json()
        post_img = data['post_img']
        img_base64 = post_img.split(',')[1]
    else:
        data = request.get_data().decode()
        temp = data.split('"')
        img_base64 = temp[3]

    # base64から画像に変換
    img_binary = base64.b64decode(img_base64)
    img = vision.Image(content=img_binary)

    response = client.document_text_detection(image=img)

    before = response.full_text_annotation.text.encode('cp932', "ignore")
    after = before.decode('cp932')

    # with open('result.jpg', "rb") as f:
    #     img_base64 = base64.b64encode(f.read()).decode('utf-8')

    response = {'result': after}
    return make_response(jsonify(response))


if __name__ == "__main__":
    app.debug = True
    app.run(host='192.168.1.186', port=5000)