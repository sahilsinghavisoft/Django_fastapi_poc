# Use the official Python image
FROM python:3.12

# Set the working directory
WORKDIR /DJANGO_FASTAPI

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 -

# Update PATH
ENV PATH="/etc/poetry/bin:$PATH"

# Copy the pyproject.toml and poetry.lock files to the working directory
COPY pyproject.toml poetry.lock ./

# Configure Poetry
RUN pip install poetry && poetry config virtualenvs.create false

# Install project dependencies
RUN poetry install --no-root --no-ansi
# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Copy the Django project files
COPY myproject/ myproject/
COPY myapp/ myapp/
COPY manage.py .
COPY main.py . 


# Copy the start.sh script to the /DJANGO_FASTAPI directory
COPY start.sh ./start.sh
# Make start.sh executable
RUN chmod +x ./start.sh
RUN python manage.py collectstatic --noinput
# Expose the ports the apps run on
EXPOSE 8000
EXPOSE 8001

# List files for debugging 
RUN ls -la /DJANGO_FASTAPI

# Command to run the start script
CMD ["./start.sh"]
