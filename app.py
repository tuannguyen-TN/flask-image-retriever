import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
import math
import random
import datetime
import sys

sys.path.append('./model/')

from retriever import Retriever

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SAVE_FOLDER = os.path.abspath('./dataset')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dqdoqdq' # required by Flask to upload images
app.config['SAVE_FOLDER'] = SAVE_FOLDER
app.config['IMAGES'] = {}
app.config['TARGET_IMAGE'] = ''
app.config['IMAGE_LIST'] = os.path.join(app.config['SAVE_FOLDER'], 'images.txt')
app.config['PREDICTIONS'] = []
app.config['MODEL_PATH'] = os.path.abspath('image-retrieval-0001/FP32/image-retrieval-0001.xml')
app.config['TOP_K'] = 0


def generate_image_list():
    with open(f'{SAVE_FOLDER}/images.txt', 'w') as image_list:
        for index, filename in enumerate(app.config['IMAGES'].keys()):
            if filename != app.config['TARGET_IMAGE']:
                line = f'{SAVE_FOLDER}/{filename} {index + 1}\n'
                image_list.write(line)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'images' not in request.files:
            flash('No image part!')
            return redirect('/')
        files = request.files.getlist('images')
        # if user does not select file, browser also
        # submit a empty part without filename
        for file in files:
            if file.filename == '':
                flash('Image not appropriate!')
                return redirect('/')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['SAVE_FOLDER'], filename))
                app.config['IMAGES'][filename] = datetime.date.today()
                if app.config['TARGET_IMAGE']:
                    generate_image_list()
                redirect('/')
    return render_template('index.html', images=app.config['IMAGES'], target_image=app.config['TARGET_IMAGE'], results=app.config['PREDICTIONS'], top_k=app.config['TOP_K'])


@app.route('/delete/<string:filename>')
def delete(filename):
    try:
        del app.config['IMAGES'][filename]
        os.remove(os.path.join(app.config['SAVE_FOLDER'], filename))
        if filename == app.config['TARGET_IMAGE'] or not app.config['IMAGE_LIST']:
            app.config['TARGET_IMAGE'] = ''
            if os.path.exists(app.config['IMAGE_LIST']):
                os.remove(app.config['IMAGE_LIST'])
        else:
            generate_image_list()
        return redirect('/')
    except:
        return 'There was a problem deleting that task!'


@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['SAVE_FOLDER'], filename)


@app.route('/select/<filename>')
def select_as_target(filename):
    app.config['TARGET_IMAGE'] = filename
    generate_image_list()
    return redirect('/')


@app.route('/deselect')
def deselect_as_target():
    app.config['TARGET_IMAGE'] = ''
    if os.path.exists(app.config['IMAGE_LIST']):
        os.remove(app.config['IMAGE_LIST'])
    return redirect('/')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    app.config['TOP_K'] = int(request.form['options'])
    if not app.config['TARGET_IMAGE']:
        if os.path.exists(app.config['IMAGE_LIST']):
            os.remove(app.config['IMAGE_LIST'])
        flash('Please choose a target image first.')
        return redirect('/')
    app.config['PREDICTIONS'] = Retriever(app.config['MODEL_PATH'], os.path.join(app.config['SAVE_FOLDER'], app.config['TARGET_IMAGE']), app.config['IMAGE_LIST'], app.config['TOP_K'])
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)