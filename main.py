# import cv2
# import numpy as np
from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
import base64

from google.cloud import vision
from google.oauth2 import service_account

from shapely.geometry import MultiPoint
from PIL import Image, ImageDraw
import io

app = Flask(__name__)
CORS(app)

# 身元証明書のjson読み込み
credentials = service_account.Credentials.from_service_account_file('./ocr-test-357422-c9b58de1a0e2.json')
client = vision.ImageAnnotatorClient(credentials=credentials)


def ocr(img_binary):
    """API使って画像でのテキストを認識"""

    img = vision.Image(content=img_binary)

    response = client.document_text_detection(image=img)

    # アノテーション結果を格納
    texts = response.text_annotations

    # 辞書としてアノテーション結果を格納
    results = []
    for text in texts:
        temp = {
            'description': text.description,
            'vertices': [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        }

        results.append(temp)

    # アノテーション結果をJSONファイルとして格納
    # with open('result.json', 'w', encoding="utf-8_sig") as json_file:
    #     json.dump(results, json_file, ensure_ascii=False, indent=2)

    return results


def merge_annotations(annotations):
    """単語を文字ブロックに直す"""

    # 全体のアノテーション結果を分割し、参考文章を取得
    sentences = annotations[0]['description'].split("\n")

    # 「今の文章、座標、文章のインデックス」と言う変数を初期化
    temp_sentence = ""
    temp_vertices = []
    cur_index = 0

    # 2番目のアノテーション結果から以下のループを行う
    results = []
    for annotation in annotations[1:]:
        # 「今の文章、座標」に、ループしているアノテーション結果（単語、座標）を後ろから追加
        temp_sentence += annotation['description']
        temp_vertices.append(annotation['vertices'])

        # もし現在の「文章インデックス」で指定する参考文章が「今の文章」とマッチすれば、文章情報を格納する
        # ここでスペースを削除する理由としては、Cloud Vision APIでスペースを入れているかどうかは抽出した単語から判断できないから
        if sentences[cur_index].replace(' ', '') == temp_sentence:
            temp_object = {
                'description': sentences[cur_index],
                'vertices': temp_vertices
            }

            results.append(temp_object)

            # 「今の文章、座標」と言う変数を初期化、「文章のインデックス」を1で足す
            temp_sentence = ""
            temp_vertices = []
            cur_index += 1

            # アノテーション結果をJSONファイルとして格納
            # with open('result_cleaned.json', 'w', encoding="utf-8_sig") as json_file:
            #     json.dump(results, json_file, ensure_ascii=False, indent=2)

    return results


def draw_boundaries(annotations, image_path, out_path):
    """認識結果を画像に描画"""
    # 凸包を取得
    convex_hulls = []
    for annotation in annotations:
        # 座標の集合を一個のリストとしてまとめる
        vertices = annotation['vertices']

        merged_vertices = []
        for vertex_set in vertices:
            for vertex in vertex_set:
                merged_vertices.append(vertex)

        # 点集合からMultiPointにして、それの凸包を求め、その凸包を構成する座標の集合を取得
        convex_hull = MultiPoint(merged_vertices).convex_hull.exterior.coords
        convex_hulls.append(list(convex_hull))

    # 各文章のバウンディングボックスを描画
    source_img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(source_img)

    for convex_hull in convex_hulls:
        draw.line(convex_hull, fill="red", width=10)

    # 描画した画像を保存
    source_img.save(out_path, "JPEG")

    return convex_hulls


def find_title(annotations, img_binary, title):
    """認識結果を画像に描画"""
    # 凸包を取得
    convex_hulls = []  # 座標
    text_blocks = []  # タイトル
    for annotation in annotations:
        # 座標の集合を一個のリストとしてまとめる
        vertices = annotation['vertices']
        text_blocks.append(annotation["description"])

        merged_vertices = []
        for vertex_set in vertices:
            for vertex in vertex_set:
                merged_vertices.append(vertex)

        # 点集合からMultiPointにして、それの凸包を求め、その凸包を構成する座標の集合を取得
        convex_hull = MultiPoint(merged_vertices).convex_hull.exterior.coords
        convex_hulls.append(list(convex_hull))

    # 各文章のバウンディングボックスを描画
    source_img = Image.open(io.BytesIO(img_binary))
    draw = ImageDraw.Draw(source_img)

    # 探しているタイトルのみ描画
    for i in range(len(text_blocks)):
        include = 0
        for w_b in list(text_blocks[i]):  # 認識した文字ブロック1文字ずつ
            for w_t in list(title):  # 探してるタイトル1文字ずつ
                if w_b == w_t:  # 同じ文字ある
                    include += 1  # カウント
        if include / len(list(title)) > 0.8:  # タイトルの文字80%以上含まれてるとき描画
            draw.line(convex_hulls[i], fill="red", width=10)
        # print(include/len(list(title)))
    # 描画した画像を保存
    # source_img.save(out_path, "JPEG")

    return source_img


@app.route("/", methods=['GET', 'POST'])
def internal_process():
    if request.is_json:
        data = request.get_json()
        post_img = data['post_img']
        img_base64 = post_img.split(',')[1]
        post_text = data['post_text']
    else:
        data = request.get_data().decode()
        temp = data.split('"')
        img_base64 = temp[3]
        post_text = temp[7]

    # base64からバイナリ画像に変換
    img_binary = base64.b64decode(img_base64)

    ocr_results = ocr(img_binary)

    anno_results = merge_annotations(ocr_results)

    highlight_img = find_title(anno_results, img_binary, post_text)

    # before = response.full_text_annotation.text.encode('cp932', "ignore")
    # after = before.decode('cp932')

    buffer = io.BytesIO()
    highlight_img.save(buffer, "jpeg")
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    response = {'result': img_base64}
    return make_response(jsonify(response))


if __name__ == "__main__":
    app.debug = True
    app.run(host='192.168.1.186', port=5000)