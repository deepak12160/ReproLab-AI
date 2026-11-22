FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    P2P_DATA_DIR=/home/user/data \
    P2P_SANDBOX=local

RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/data \
    && chown -R user:user /home/user

WORKDIR /home/user/app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend backend
COPY frontend frontend
COPY sample_papers sample_papers

USER user

WORKDIR /home/user/app/backend
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
