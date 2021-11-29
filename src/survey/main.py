import os
import random

from databases import Database

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import starlette.status as status

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

pictures: list[str] = []

database = Database('sqlite:///results.db')


@app.on_event("startup")
async def startup():

    await database.connect()
    query = "select 1 from results"
    await database.execute(query=query)

    global pictures
    pictures = [x for x in os.listdir("static") if x != ".gitkeep"]
    print(pictures)


def load_picture():
    rpicture = random.choice(pictures)
    if not os.path.exists(os.path.join("static", rpicture)):
        return None

    return rpicture


@app.get("/", response_class=HTMLResponse)
async def show_picture(request: Request):
    result = load_picture()
    if result is None:
        raise HTTPException(status_code=500, detail="Internal server error: picture not found")

    return templates.TemplateResponse(
        "picture_shower.html",
        {
            'request': request,
            "picture": result,
        }
    )


@app.post("/submit")
async def submit(request: Request, answer: str = Form(...), key: str = Form(...)):
    query = "insert into results(ipport, key, answer) VALUES (:ipport, :key, :answer)"
    values = {
        "ipport": f"{request.client.host}_{request.client.port}",
        "key": key,
        "answer": answer
    }

    await database.execute(query=query, values=values)
    return RedirectResponse(url='/', status_code=status.HTTP_303_SEE_OTHER)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")
