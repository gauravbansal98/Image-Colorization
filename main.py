import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from data import colorize_image as CI
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Initialize colorization class
colorModel = CI.ColorizeImageTorch(Xd=256)

# Load the model
colorModel.prep_net(None,'models/pytorch/caffemodel.pth')
mask = np.zeros((1,256,256)) # giving no user points, so mask is all 0's
input_ab = np.zeros((2,256,256)) # ab values of user points, default to 0 for no input

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # filename = "testing.jpg"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        colorModel.load_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img_out = colorModel.net_forward(input_ab,mask) # run model, returns 256x256 image
        img_gray_fullres = colorModel.get_img_gray_fullres() # get grayscale image at fullresolution
        img_out_fullres = colorModel.get_img_fullres() # get image at full resolution
        im_BW = Image.fromarray(img_gray_fullres)
        im_color = Image.fromarray(img_out_fullres)
        #im_BW.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        im_color.save(os.path.join(app.config['UPLOAD_FOLDER'], filename.split('.')[0] + "1.jpeg"))
        flash('Image successfully uploaded and displayed')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == "__main__":
    app.run()
