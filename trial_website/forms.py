from flask_wtf import FlaskForm
from wtforms import SubmitField


class IndexPageForm(FlaskForm):
    submit = SubmitField("Upload a video")
