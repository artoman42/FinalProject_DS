FROM python:3.10

ARG model_name=prod_model.pkl
ARG vectorizer_name=prod_vectorizer.pkl
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

WORKDIR /app

COPY src/data_process/ /app/src/data_process

COPY src/inference/ /app/src/inference

COPY src/utils.py /app/src/

COPY src/${CONF_PATH} /app/src/

COPY outputs/models/${model_name} /app/outputs/models/${model_name}

COPY outputs/vectorizers/${vectorizer_name} /app/outputs/vectorizers/${vectorizer_name}

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD python3 /app/src/data_process/data_loader.py --mode inference && \
    python3 /app/src/data_process/data_processor.py --mode inference && \
    python3 /app/src/inference/inference.py
