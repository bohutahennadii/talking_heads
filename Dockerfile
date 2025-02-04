FROM python:3.10

EXPOSE 8080

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN python -m pip install -r requirements.txt

WORKDIR /app

RUN python -m spacy download en_core_web_sm
RUN pip install datasets transformers[sentencepiece]
RUN mkdir -p models/result

COPY . /app

CMD ["python", "main.py", "--reload"]