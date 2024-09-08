from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from utils import load_artifacts, predict

# Create a FastAPI app instance
app = FastAPI()

# Allow Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Load the artifacts once when the server starts
pipeline, scaler, model = load_artifacts()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_credit(request: Request):
    try:
        # Get JSON data from the request
        data = await request.json()
        
        # Check if data is not empty
        if not data:
            raise HTTPException(status_code=400, detail="No data provided")

        formatted_data = {key: [value] if not isinstance(value, list) else value for key, value in data.items()}
        
        # Use the predict function from utils to get predictions
        prediction, probability = predict(formatted_data, pipeline, scaler, model)
        print(prediction, probability)

        response = {
            "Predicted Labels": int(prediction),
            "Predicted Probabilities": float(probability)
        }
        
        print(response)
        # Return the prediction results as a JSON response
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5005)
