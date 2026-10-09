# Signal Intelligence demo app — self-contained Streamlit image.
# The demo dataset is baked in (app/demo/), so the container needs no
# credentials and makes no network calls at runtime.

FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY .streamlit/ .streamlit/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/_stcore/health')"

# CORS/XSRF disabled and WS compression off: standard settings for running
# Streamlit behind a reverse proxy (Lightsail's load balancer), whose
# forwarded Origin/Host headers would otherwise trip Tornado's checks.
# Safe here: read-only app, no forms, no user data.
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8080", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false", \
     "--server.enableWebsocketCompression=false"]
