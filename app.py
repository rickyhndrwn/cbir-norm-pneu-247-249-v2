from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for
from pathlib import Path
from random import shuffle
from tensorflow.keras.models import load_model
import numpy as np
import os
import sys

from feature_extractor import FeatureExtractor

app = Flask(__name__)

db_image_dir = './static/db_image'
db_feature_dir = './static/db_feature'
db_model_dir = './static/db_model'
db_test_image_dir = './static/db_test_image'

fe_dn201 = FeatureExtractor(load_model(db_model_dir + '/cxr_dn201.h5', compile=False))
features_dn201 = np.load(db_feature_dir + '/features_dn201.npy')

db_img_paths = list()
for img_path in sorted(Path(db_image_dir).glob("*.jpeg")):
    db_img_paths.append(img_path)

db_test_image_paths = list()
for img_path in sorted(Path(db_test_image_dir).glob("*.jpeg")):
    db_test_image_paths.append(img_path)

def find_y_pred(scores):
    y_pred = list()
    for score in scores:
        db_img_name = os.path.basename(score[1])
        y_pred.append(db_img_name[0])
    
    return np.array(y_pred, dtype=int)

@app.route('/', methods=['GET', 'POST'])
def index():
    shuffle(db_test_image_paths)
    random_test_image_dir = db_test_image_paths[:9]
    random_test_image_name = [img_dir.name for img_dir in random_test_image_dir]
    if request.method == 'POST':
        query_img_name = ''
        try:
            file = request.files['query_img']
            img = Image.open(file.stream)  # PIL image
            query_img_name = file.filename
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        except:
            filename = request.form.get('query_img')
            img = Image.open(filename)
            query_img_name = filename.split('/')[-1]
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + query_img_name

        query_img_class = query_img_name[0]

        # Save query image
        img.save(uploaded_img_path)
        uploaded_img_path = '../' + uploaded_img_path

        # Run search
        query = fe_dn201.extract(img)
        feature_difference = features_dn201 - query
        dists = np.linalg.norm(feature_difference, axis=1)  # L2 distances to features
        n_img = 25
        ids = np.argsort(dists)[:n_img]  # Top n_img results
        scores = [(dists[id], db_img_paths[id]) for id in ids]
        
        # Evaluate result
        y_true = np.array([query_img_class for _ in range(n_img)], dtype=int)
        y_pred = find_y_pred(scores)
        relevant_img = np.sum(y_true == y_pred)
        result_eval = f"{relevant_img}/{n_img}"

        return render_template('index.html',
                               test_images=random_test_image_dir,
                               test_images_names=random_test_image_name,
                               query_path=uploaded_img_path,
                               query_image_name=query_img_name,
                               scores=scores,
                               eval_score=result_eval)
    else:
        return render_template('index.html',
                               test_images=random_test_image_dir,
                               test_images_names=random_test_image_name,
                               )

if __name__ == '__main__':
    app.run(debug=True)
