# 1. Base Image: Python 3.10 (Slim version for speed)
FROM python:3.10-slim

# 2. Set Environment Variables
# Ensures logs stream directly to Google Cloud Console
ENV PYTHONUNBUFFERED=1
# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Set a dummy secret key for the build phase (fix for collectstatic)
# This passes the dummy key ONLY during the command execution
RUN DJANGO_SECRET_KEY=build_dummy_key python src/portal/manage.py collectstatic --noinput

# 3. Set Working Directory
WORKDIR /app

# 4. Install System Dependencies (Needed for PostgreSQL/BigQuery)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python Dependencies
# We copy ONLY requirements.txt first to leverage Docker cache
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the Entire Project
COPY . /app/

# 7. Collect Static Files
# This gathers all CSS/JS into one folder for the web server
RUN python src/portal/manage.py collectstatic --noinput --clear

# 8. Expose the Port (Google Cloud expects 8080)
EXPOSE 8080

# Collect static files
RUN python src/portal/manage.py collectstatic --noinput

# 9. Start the Server (The "EntryPoint")
# --chdir src/portal: Tells Gunicorn to look inside the portal folder
# config.wsgi:application: Points to the WSGI file
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--chdir", "src/portal", "config.wsgi:application"]