# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:56:23 2019

@author: Aymar
"""

#import os
from flask import Flask, render_template, request, url_for, redirect


app = Flask(__name__)

# Base endpoint to perform prediction.
@app.route('/', methods=['GET'])
def load():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
#        arr = request.get_data().decode('utf8')
#        print(len(arr))
        app.post('index.html')
    return render_template('SignUp.html')

@app.route('/index', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('SignUp'))
    return render_template('index.html')
    
@app.route('/MainPage', methods=['GET','POST'])
def MainPage():
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/SignUp',methods=['GET','POST'])
def SignUp():
    if request.method == 'POST':
        return render_template('index.html')#Changed this
    return render_template('SignUp.html')
@app.route('/ContactPage.html',methods = ['GET'])
def ContactPage():
    return render_template('ContactPage.html')

#@app.route('/cyberfooter.jpg',methods = ['GET'])
#def cyberfooter():
#    return render_template('cyberfooter.jpg',encoding="utf8")

if __name__ == '__main__':
    app.run(debug=True)
