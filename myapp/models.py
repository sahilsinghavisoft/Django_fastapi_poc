from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    is_student = models.BooleanField(default=False)
    is_teacher = models.BooleanField(default=False)


    def save(self, *args, **kwargs):
        # Ensure only one of these flags is True
        if self.is_student and self.is_teacher:
            raise ValueError("A user cannot be both a student and a teacher.")
        super().save(*args, **kwargs)

