FROM python:3.9

WORKDIR /the/workdir/path

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY ./src ./src

EXPOSE 80

CMD [uvicorn src.main:app -p 80:80 --reload ]
