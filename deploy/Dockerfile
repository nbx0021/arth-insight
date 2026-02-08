# 1. Base Image
FROM python:3.10-slim

# 2. Set Environment Variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 3. Set Working Directory
WORKDIR /app

# 4. Install System Dependencies (gcc is needed for some python packages)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python Dependencies
# We do this BEFORE copying the whole project to save time (Docker Cache)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the Entire Project
# Now your code actually exists inside the container
COPY . /app/

# 7. Collect Static Files
# We run this NOW because the code and libraries are finally present.
# We use the absolute path /app/src/portal/manage.py
RUN DJANGO_SECRET_KEY=build_dummy_key \
    PYTHONPATH=/app/src/portal \
    python /app/src/portal/manage.py collectstatic --noinput --clear

# 8. Expose the Port
EXPOSE 8080

# 9. Start the Server
# We use --chdir so Gunicorn starts inside the portal folder
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--chdir", "src/portal", "config.wsgi:application"]