from fastapi import FastAPI

from deploy.routers import captcha_translate

app = FastAPI()

app.include_router(captcha_translate.router)

@app.get("/")
def home_page():
    return {"Service": "Home Page"}