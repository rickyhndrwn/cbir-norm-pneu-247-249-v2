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


@app.route('/', methods=['GET', 'POST'])
def index():
    db_model_dict, db_feature_dict, db_img_paths, random_test_image_dir, random_test_image_name = init_db_values_func()
    
    if request.method == 'POST':
        try:
            file = request.files['query_img']
            query_img = Image.open(file.stream)  # PIL image
            query_img_name = file.filename
            
        except:
            filename = request.form.get('query_img')
            query_img = Image.open(filename)
            query_img_name = filename.split('/')[-1]

        query_img_class = query_img_name[0]

        # Save query image
        uploaded_img_path = "static/uploaded/temp" + os.path.splitext(query_img_name)[1]
        query_img.save(uploaded_img_path)
        uploaded_img_path = '../' + uploaded_img_path

        # Run search
        n_img = 25
        y_true = np.array([query_img_class for _ in range(n_img)], dtype=int)

        score_dict = dict()
        relevant_img_count_dict = dict()
        for (db_model_dict_key, db_model_dict_value), extracted_features in zip(db_model_dict.items(), db_feature_dict.values()):
            scores = retrieve_images_func(db_model_dict_value, extracted_features, db_img_paths, query_img, n_img)
            
            y_pred = find_y_pred_func(scores)
            relevant_img_count = np.sum(y_true == y_pred)
            
            score_dict[db_model_dict_key] = scores
            relevant_img_count_dict[db_model_dict_key] = relevant_img_count
        
        # Get key with best result
        best_result_key = max(relevant_img_count_dict, key=relevant_img_count_dict.get)
        result_eval = f"{relevant_img_count_dict[best_result_key]}/{n_img}"

        return render_template('index.html',
                               test_images=random_test_image_dir,
                               test_images_names=random_test_image_name,
                               query_path=uploaded_img_path,
                               query_image_name=query_img_name,
                               best_model=best_result_key,
                               scores=score_dict[best_result_key],
                               eval_score=result_eval)
    
    else:
        return render_template('index.html',
                               test_images=random_test_image_dir,
                               test_images_names=random_test_image_name,
                               )


def init_db_values_func():
    db_image_dir = './static/db_image'
    db_feature_dir = './static/db_feature'
    db_model_dir = './static/db_model'
    db_test_image_dir = './static/db_test_image'

    db_model_dict = {
        # 'cxr_dn121': db_model_dir + '/cxr_dn121.h5',
        'cxr_dn169': db_model_dir + '/cxr_dn169.h5',
        'cxr_dn201': db_model_dir + '/cxr_dn201.h5',
    }

    db_feature_dict = {
        # 'features_dn121': db_feature_dir + '/features_dn121.npy',
        'features_dn169': db_feature_dir + '/features_dn169.npy',
        'features_dn201': db_feature_dir + '/features_dn201.npy',
    }

    db_img_paths = list()
    for img_path in sorted(Path(db_image_dir).glob("*.jpeg")):
        db_img_paths.append(img_path)

    db_test_image_paths = list()
    for img_path in sorted(Path(db_test_image_dir).glob("*.jpeg")):
        db_test_image_paths.append(img_path)
    
    shuffle(db_test_image_paths)
    random_test_image_dir = db_test_image_paths[:9]
    random_test_image_name = [img_dir.name for img_dir in random_test_image_dir]
    return db_model_dict,db_feature_dict,db_img_paths,random_test_image_dir,random_test_image_name


def find_y_pred_func(scores):
    y_pred = list()
    for score in scores:
        db_img_name = os.path.basename(score[1])
        y_pred.append(db_img_name[0])
    
    return np.array(y_pred, dtype=int)


def retrieve_images_func(extractor_dir, features_dir, db_img_paths, query_img, n_img):
    extractor_class = FeatureExtractor(load_model(extractor_dir, compile=False))
    features = np.load(features_dir)
    
    query = extractor_class.extract(query_img)
    feature_difference = features - query
    
    dists = np.linalg.norm(feature_difference, axis=1)  # L2 distances to features
    ids = np.argsort(dists)[:n_img]  # Top n_img results
    scores = [(dists[id], db_img_paths[id]) for id in ids]
    
    return scores


if __name__ == '__main__':
    app.run()
