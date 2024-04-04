from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('handwritten_digits.model')
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/submit_pdf', methods=['POST'])
def submit_pdf():
    if request.method == 'POST':
        img = request.files['pdf_file']
        if img:
            try:
                img = cv2.imread(img)
                img = np.invert(np.array([img]))
                prediction = model.predict(img)
                print("The number is probably a {}".format(np.argmax(prediction)))
                plt.imshow(img[0], cmap=plt.cm.binary)
                plt.show()
                
            except:
                print("Error reading image! Proceeding with next image...")


if __name__ == '__main__':
    app.run(debug=True)  # Remove for production
