import streamlit as st

st.title = "TalkingHeads"



#import os

#from fastapi import FastAPI, File, Form, Request, UploadFile, status
#from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
#rom fastapi.templating import Jinja2Templates

#from pydantic import BaseModel

#os.system("python -m spacy download en_core_web_sm")

#from model_ml import (
#    get_answer
#)


#class TextInput(BaseModel):
#    text: str

#app = FastAPI()

#templates = Jinja2Templates(directory="templates")

###
# "/talking-heads" in route used for compatibles with Prod
###

#@app.get("/", response_class=HTMLResponse)
#async def read_root(request: Request):
#    return templates.TemplateResponse("index.html", {"request": request})


#@app.post("/process-question/")
#@app.post("/talking-heads/process-question/")
#async def upload_text(text: TextInput):
#    result =  get_answer(text.text)
#    return {"answer": result.result}

#if __name__ == "__main__":
#    import uvicorn
#
#    uvicorn.run(app, host="0.0.0.0", port=8080)
#