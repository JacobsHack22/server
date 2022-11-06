from flask import Flask, request
from flask import send_file
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from PIL import Image

def create_app() -> Flask:
    app = Flask(__name__)

    img_model = SentenceTransformer('clip-ViT-B-32')
    corpus_emb = np.load('model/embeddings.npy')

    # @app.route('/tree_image', methods=['GET'])
    # def add_imag():
    #     return send_file('tree.png', mimetype='image/png') 
        
    @app.route('/recongnize_tree', methods=['GET'])
    def recongnize_tree():
        imagefile = request.files.get('photo', '')
        img_np = to_gray_contrast(imagefile.read())
        tree_type = find_nearest(img_np)
        return {'tree_type': tree_type}, 200

    def to_gray_contrast(str):
        nparr = np.fromstring(str, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        img_np = cv2.resize(img_np, (224, 224))
        blur = cv2.GaussianBlur(img_np, (5,5),0)
        blur = blur.astype('uint8')
        _, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)        
        kernel = np.ones((3,3), np.uint8)
        closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
        img = closing.astype('float32')
        #plt.imsave('govno_lolo.jpg', img, cmap='gray')
        return img

    def find_nearest(img_np):
        img = Image.fromarray(img_np)
        query = img_model.encode(img, convert_to_tensor=True)
        lst = util.semantic_search(query_embeddings=query, corpus_embeddings=corpus_emb, top_k=50)
        dicts = lst[0]
        cnt = [0] * 10
        for dic in dicts:
            print(str(dic['corpus_id']) + '   ' + str(dic['score']))
            cnt[int(dic['corpus_id'] / 75)] += 1
        max_index = cnt.index(max(cnt))
        return map_to_tree(max_index)
        
    def map_to_tree(id):
        if id == 0:
            return 'Acer'
        elif id == 1:
            return 'Quercus'
        elif id == 2:
            return 'Betula'
        elif id == 3:
            return 'Sorbus'
        elif id == 4:
            return 'Populus'
        elif id == 5:
            return 'Ulmus'
        elif id == 6:
            return 'Tilia'
        elif id == 7:
            return 'Fagus'
        elif id == 8:
            return 'Salix'
        elif id == 9:
            return 'Alnus'


    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
