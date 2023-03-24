TEMPLATES = [
    {
        # Template backend to be used, For example Jinja
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        # Directories for templates
        "DIRS": [],
        "APP_DIRS": True,
        # options to configure
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]
