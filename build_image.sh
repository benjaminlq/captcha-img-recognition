#!/bin/bash

docker build -t captcha_ctc  .
docker tag captcha_ctc lequan2902/captcha_ctc:latest
docker push lequan2902/captcha_ctc:latest