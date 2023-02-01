from fastapi import FastAPI, Request
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import matplotlib
import matplotlib.pyplot as plt

def create_model():
    mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
    mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
    X, y = mnist['data'], mnist['target']
    
    X = X.to_numpy()
    y = y.to_numpy()

    # przetasowanie zbioru uczącego
    shuffle_index = np.random.permutation(60000)
    X, y = X[shuffle_index], y[shuffle_index]

    forest_clf = RandomForestClassifier(random_state = 42)
    forest_clf.fit(X, y)

    return forest_clf


app = FastAPI()
templates = Jinja2Templates(directory="")
forest_clf = create_model()


@app.post("/api")
async def main(array: list):
    digit = np.asarray(array)

    # digit_image = digit.reshape(28,28)
    # plt.imshow(digit_image, cmap = matplotlib.cm.binary)
    # plt.axis("off")
    # plt.show()

    response = forest_clf.predict([digit])
    print(f'response: {response[0]}')
    return {"response": str(response[0])}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})