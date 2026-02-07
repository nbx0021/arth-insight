import os
from dotenv import load_dotenv
import json
from pathlib import Path



# 1. BASE PATHS
# Points to 'src/portal'
BASE_DIR = Path(__file__).resolve().parent.parent
# Points to the absolute root 'arth-insight'
ROOT_DIR = BASE_DIR.parent.parent

# --- SMART CREDENTIALS SETUP ---
GOOGLE_CREDENTIALS_JSON = os.getenv('GCP_CREDENTIALS_JSON')

if GOOGLE_CREDENTIALS_JSON:
    # 1. Determine where we are (Cloud or Laptop?)
    if os.path.exists('/app'):
        # We are on Render (Docker) -> Use the Cloud Folder
        PROJECT_ROOT = Path('/app')
    else:
        # We are on Laptop -> Go up 4 levels to find 'arth-insight' root
        # config -> portal -> src -> arth-insight
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

    # 2. Define the path for the key file
    cred_file_path = PROJECT_ROOT / 'service-account.json'
    
    # 3. Write the secret key to that file
    with open(cred_file_path, 'w') as f:
        f.write(GOOGLE_CREDENTIALS_JSON)
    
    # 4. Tell Google libraries to look there
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(cred_file_path)
    
    # Debug print (Optional: helps you see where it saved)
    print(f"🔑 Google Credentials saved to: {cred_file_path}")
# -------------------------------
# 2. LOAD SECRETS FROM .ENV
load_dotenv(os.path.join(ROOT_DIR, '.env'))

# 3. SECURITY
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-fallback-key-123')
DEBUG = os.getenv('DEBUG', 'True') == 'True'
ALLOWED_HOSTS = ['*'] # For development only

# 4. APP DEFINITION
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third Party
    'rest_framework',
    
    # Your Apps
    'dashboard',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    "whitenoise.middleware.WhiteNoiseMiddleware",
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'], # This tells Django to look in src/portal/templates
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

# 5. DATABASE (Internal Django DB)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# 6. INTERNATIONALIZATION
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata' # Set for India
USE_I18N = True
USE_TZ = True

# 7. STATIC AND MEDIA
STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / "static"]
MEDIA_URL = 'media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
# BASE_DIR is 'src/portal'
BASE_DIR = Path(__file__).resolve().parent.parent


# --- STATIC FILES CONFIGURATION ---
STATIC_URL = '/static/'

# 1. THE DESTINATION (For Production)
# When we run 'collectstatic', all files move here.
# We name it 'staticfiles' to avoid conflict with your source folder 'static'.
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# 2. THE SOURCE (Where you put your CSS/JS/Images during development)
# This tells Django to look in 'src/portal/static'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

# 3. WHITE NOISE (The Server)
# Compresses files for speed on Render.
if not DEBUG:
    STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# CACHING CONFIGURATION (RAM Cache)
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
    }
}

ALLOWED_HOSTS = ['*']

# Add this line so the app knows which dataset to use!
BIGQUERY_DATASET = os.getenv('BIGQUERY_DATASET', 'stock_raw_data')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'