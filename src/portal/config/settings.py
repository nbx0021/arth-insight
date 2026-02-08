import os
import json
from pathlib import Path
from dotenv import load_dotenv

# 1. BASE PATHS
# Path to 'src/portal' (where manage.py lives)
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to the absolute root 'arth-insight'
ROOT_DIR = BASE_DIR.parent.parent

# 2. LOAD SECRETS
# Load .env from the root directory
load_dotenv(os.path.join(ROOT_DIR, '.env'))

SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-build-key-123')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = ['*']

# --- GOOGLE CLOUD / BIGQUERY SETUP ---
GOOGLE_CREDENTIALS_JSON = os.getenv('GCP_CREDENTIALS_JSON')

if GOOGLE_CREDENTIALS_JSON:
    # Determine the project root for the key file
    # Docker uses '/app', Local uses ROOT_DIR
    project_root = Path('/app') if os.path.exists('/app') else ROOT_DIR
    cred_file_path = project_root / 'service-account.json'
    
    # Write credentials only if the file doesn't exist to save IO
    if not cred_file_path.exists():
        with open(cred_file_path, 'w') as f:
            f.write(GOOGLE_CREDENTIALS_JSON)
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(cred_file_path)

BIGQUERY_DATASET = os.getenv('GCP_DATASET_ID', 'stock_raw_data')

# 3. APP DEFINITION
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third Party
    'rest_framework',
    'whitenoise.runserver_nostatic', # Optional: makes whitenoise work in dev
    
    # Your Apps
    'dashboard',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # MUST BE HERE
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# 4. DATABASE
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# 5. STATIC FILES
STATIC_URL = '/static/'

# Where files are collected FOR production (Render/Docker)
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Where you put your source files DURING development
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]

# Optimize static files for production
if not DEBUG:
    # Compresses and hashes files (e.g., style.css -> style.a1b2c3.css)
    STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# 6. CACHING (In-Memory)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
    }
}

# 7. OTHER SETTINGS
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'