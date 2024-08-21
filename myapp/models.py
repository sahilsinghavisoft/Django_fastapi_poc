from djongo import models
from bson import ObjectId
from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError

class User(AbstractUser):
    _id = models.ObjectIdField(primary_key=True, default=ObjectId, editable=False)
    is_student = models.BooleanField(default=False)
    is_teacher = models.BooleanField(default=False)

    def clean(self):
        # Ensure only one of these flags is True
        if self.is_student and self.is_teacher:
            raise ValidationError("A user cannot be both a student and a teacher.")
        super().clean()

    def save(self, *args, **kwargs):
        # Call clean() to validate model before saving
        self.clean()
        super().save(*args, **kwargs)
