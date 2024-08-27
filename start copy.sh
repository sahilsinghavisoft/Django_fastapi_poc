#!/bin/bash
python manage.py collectstatic --noinput

python manage.py makemigrations
python manage.py migrate
# Start Django server using Gunicorn
echo "Starting Django..."
poetry run uvicorn myproject.asgi:application --host 0.0.0.0 --port 8000 &
# Create a superuser if not already create
# Start FastAPI server
echo "Creating Django superuser..."
DJANGO_SUPERUSER_USERNAME=sahil123
DJANGO_SUPERUSER_PASSWORD=sahil123
DJANGO_SUPERUSER_EMAIL=sahil@example.com

python manage.py shell << END
from django.contrib.auth import get_user_model
User = get_user_model()
if not User.objects.filter(username='${DJANGO_SUPERUSER_USERNAME}').exists():
    User.objects.create_superuser('${DJANGO_SUPERUSER_USERNAME}', '${DJANGO_SUPERUSER_EMAIL}', '${DJANGO_SUPERUSER_PASSWORD}')
    print("Superuser created.") 
else:
    print("Superuser already exists.")
END
echo "Starting FastAPI..."
poetry run uvicorn main:app --host 0.0.0.0 --port 8001
