FROM python:3.9

WORKDIR /captcha

COPY ./requirements-deploy.txt ./setup.py ./setup.cfg ./

RUN pip install --upgrade pip
RUN pip install -r requirements-deploy.txt --no-cache-dir

COPY ./src ./src
COPY ./artifacts ./artifacts

RUN pip install -e .

EXPOSE 80

CMD ["uvicorn", "src.deploy.main:app", "--host", "0.0.0.0", "--port", "80"]
