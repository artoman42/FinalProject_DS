FROM python:3.9

ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

WORKDIR /app

COPY src/data_process/ /app/src/data_process

COPY src/train/ /app/src/train

COPY src/utils.py /app/src/

COPY src/${CONF_PATH} /app/src/

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

CMD python3 /app/src/data_process/data_loader.py --mode training && \
    python3 /app/src/data_process/data_processor.py --mode training && \
    python3 /app/src/train/train.py
