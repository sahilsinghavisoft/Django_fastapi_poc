from mongoengine import Document, StringField, FileField, DateTimeField, IntField
from datetime import datetime, timezone

# Define the modified TeacherUpload model
class TeacherUpload(Document):
    user_id = IntField(required=True)
    teacher = StringField(required=True)
    pdf_file = FileField(required=True)
    uploaded_at = DateTimeField(default=datetime.now(timezone.utc))

    def __str__(self):
        return f"{self.teacher_name}'s upload: {self.pdf_file.name}"

class Question(Document):
    user_id=IntField(required=True)
    student = StringField(required=True)
    question_text = StringField(required=True)
    answer_text = StringField(blank=True, null=True)
    created_at = DateTimeField(default=datetime.now(timezone.utc))

    def __str__(self):
        return f"{self.student.username}'s question: {self.question_text[:50]}..."