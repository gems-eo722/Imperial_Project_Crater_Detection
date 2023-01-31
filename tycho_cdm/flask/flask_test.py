from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    # do something with the file
    return 'File uploaded successfully'

if __name__ == '__main__':
    app.run()