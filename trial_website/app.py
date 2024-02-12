import os

import dotenv
from flask import Flask, render_template, redirect, url_for

from forms import IndexPageForm
from config import Config

dotenv.load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config.from_object(Config)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = IndexPageForm()
    if form.validate_on_submit():
        return redirect(url_for("upload_video"))

    return render_template("index.html", form=form)


@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    # video = request.files
    return render_template("video_upload.html")


if __name__ == '__main__':
    app.run(port=5500, debug=True, use_reloader=True, threaded=True)
