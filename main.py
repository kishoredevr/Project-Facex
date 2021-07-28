
from flask import Flask, render_template, Response
from camera import VideoCamera



app = Flask(__name__, static_folder='static')
#cors = CORS(app, resources={r"/foo": {"origins": "*"}})
#app.config['CORS_HEADERS'] = 'Content-Type'
#app.config['CORS_HEADERS'] = 'Access-Control-Allow-Origin'
#CORS(app)

UPLOAD_FOLDER = 'static'

app.config['UPLOAD_PATH'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('homenew.html')

@app.route('/cam')
def cam():
    return render_template('camera.html')


@app.route("/test")
def test():
    return "<h1>APP API Is Working check for other servicess --KPS004</h1>"

def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True,port=5002)
