FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc g++ make curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-télécharge cmdstan (backend Stan de Prophet) pour éviter le download au premier run
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

COPY src/ src/
COPY *.py .

# data/ et logs/ montés en volume (persistance hors conteneur)
VOLUME ["/app/data", "/app/logs"]

CMD ["python", "run_weekly.py"]
