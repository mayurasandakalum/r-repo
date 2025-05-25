from flask_wtf import FlaskForm
from flask import request  # Add this import
from wtforms import (
    StringField,
    PasswordField,
    SubmitField,
    BooleanField,
    EmailField,
    RadioField,
    SelectField,
    TextAreaField,
    IntegerField,
    FloatField,
    FieldList,
    FormField,
)
from wtforms.validators import (
    DataRequired,
    Email,
    EqualTo,
    Optional,
    NumberRange,
    ValidationError,
)
from .models import Subject  # Add this import


# Remove UserLoginForm and RegistrationForm as they won't be needed anymore

class QuizForm(FlaskForm):
    choice_pk = RadioField("Choice", validators=[DataRequired()])


class SubQuestionForm(FlaskForm):
    text = TextAreaField("Sub-Question Text", validators=[DataRequired()])
    instructions = TextAreaField("Instructions")
    correct_answer = StringField("Correct Answer", validators=[DataRequired()])
    answer_type = SelectField(
        "Answer Type",
        choices=[("number", "Number"), ("time", "Time")],
        validators=[DataRequired()],
    )
    min_value = FloatField("Minimum Value", validators=[Optional()])
    max_value = FloatField("Maximum Value", validators=[Optional()])
    time_format = StringField("Time Format")
    difficulty_level = IntegerField(
        "Difficulty Level", validators=[NumberRange(min=1, max=5)], default=1
    )
    points = IntegerField("Points", validators=[NumberRange(min=1)], default=1)
    hint = TextAreaField("Hint")
    submit = SubmitField("Save Sub-Question")


class QuestionForm(FlaskForm):
    text = TextAreaField("Question Text", validators=[DataRequired()])
    subject = SelectField(
        "Subject",
        choices=Subject.CHOICES,
        validators=[DataRequired()],
    )
    answer_method = SelectField(
        "Answer Method",
        choices=[],  # Will be populated based on subject
        validators=[DataRequired()],
    )
    is_published = BooleanField("Published", default=True)  # Set default to True
    sub_questions = FieldList(FormField(SubQuestionForm), min_entries=1)
    submit = SubmitField("Save Question")

    def __init__(self, *args, **kwargs):
        super(QuestionForm, self).__init__(*args, **kwargs)

        # Set initial subject from kwargs or use ADDITION as default
        if kwargs.get("obj"):
            subject = kwargs["obj"].subject
        else:
            subject = kwargs.get("initial_subject", Subject.ADDITION)
            self.subject.data = subject

        # Set the answer method choices based on subject
        self.answer_method.choices = Subject.ANSWER_METHODS.get(subject, [])

    def validate_answer_method(self, field):
        # Ensure the selected answer method is valid for the chosen subject
        if field.data not in [
            method[0] for method in Subject.ANSWER_METHODS.get(self.subject.data, [])
        ]:
            raise ValidationError("Invalid answer method for selected subject")
