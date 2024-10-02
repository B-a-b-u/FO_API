# Fertilizer Optimizer Backend (FastAPI)

### Sustainable Fertilizer Usage Optimizer for Higher Yield

This is the backend service for the **Fertilizer Optimizer** project, part of the **Smart India Hackathon 2024**. It is developed using **FastAPI** to serve API requests for calculating the optimal fertilizer type and amount based on soil data, crop type, and weather conditions.

## Features

- **REST API** built with FastAPI.
- **Fertilizer recommendation** based on a trained neural network model.
- **Integration with location and weather APIs** for real-time data.
  
## Tech Stack

- **FastAPI**: High-performance backend framework for building APIs.
- **Python**: Core programming language for model integration.
- **Neural Network**: Deep learning model used for fertilizer recommendations.
- **Uvicorn**: ASGI server for running the FastAPI app.

## Installation and Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/B-a-b-u/FO_API.git
    cd FO_API
    ```

2. Set up a virtual environment and activate it:
    ```bash
    python -m venv environment_name
    source environment_name/bin/activate  # For Windows use `environment\Scripts\activate`
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the FastAPI server:
    ```bash
    uvicorn main:app --reload
    ```

5. The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

- `POST /ferti-predict`: Takes input data (soil health, crop type, weather) and returns fertilizer type .
- `GET /`: Gives a message for checking the API is working fine.
  
## Hosted API
- This FastAPI Backend is hosted on [Render](https://render.com/)
  - [FO_API](https://fo-api.onrender.com)
 
## Front-end and Model Code
- Front-end is developed with React Native on my [Repo](https://github.com/B-a-b-u/Fertilizer_Frontend)
- Model is developed with Classification Neural Network in [Colab](https://colab.research.google.com/drive/119ifp2jShgDod6BZxuQdMUzsVxNImcbP?usp=sharing).
