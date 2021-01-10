import uuid
import time
import joblib
import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse

app_version = "2.0.0"
app = FastAPI(
    title="Activity Recognition",
    version=app_version,
    description=
    "An micro-service for classifiying the Activity based on data from Accelerometer",
    docs_url="/docs",
    redoc_url="/redoc")

categories = {
    0: "Working at Computer",
    1: "Standing Up, Walking and Going Up\Down Stairs",
    2: "Standing",
    3: "Walking",
    4: "Going Up\Down Stairs",
    5: "Walking and Talking with Someone",
    6: "Talking while Standing"
}


# Routes
@app.get('/', include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


@app.get('/api/v1/predict')
async def predict(x, y, z):
    start_ms = int(round(time.time() * 1000))
    if x and y and z:
        model = joblib.load("weights.joblib")
        return {
            "meta": {
                "x": x,
                "y": y,
                "z": z,
                "time_ms": int(round(time.time() * 1000)) - start_ms,
                "status": "success",
                "job_id": str(uuid.uuid1())
            },
            "data": {
                "activity": str(categories[int(model.predict([[x, y, z]]))]),
                "label": int(model.predict([[x, y, z]]))
            }
        }
    else:
        return {
            "meta": {
                "x": x,
                "y": y,
                "z": z,
                "time_ms": int(round(time.time() * 1000)) - start_ms,
                "status": "failure",
                "job_id": str(uuid.uuid1())
            },
            "data": {
                "activity": None,
                "label": None
            }
        }


def vicara_openapi():

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Activity Recognition",
        version=app_version,
        description=
        "An micro-service for classifiying the Activity based on data from Accelerometer",
        routes=app.routes)

    openapi_schema["info"]["x-logo"] = {
        "url":
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS7BuMiEr54Mh-wTAW_dJlBzah4o89hPqkufw&usqp=CAU"
    }
    app.openapi_schema = openapi_schema

    return app.openapi_schema


app.openapi = vicara_openapi

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port="8000")
