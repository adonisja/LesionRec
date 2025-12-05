# Deployment Guide for LesionRec (Lumina)

This guide covers how to deploy the Lumina application to a production environment. The application consists of two parts:
1.  **Backend**: FastAPI (Python)
2.  **Frontend**: React + Vite (TypeScript)

## Prerequisites

-   GitHub Account (for connecting to deployment services)
-   [Render](https://render.com/) Account (for Backend)
-   [Vercel](https://vercel.com/) Account (for Frontend)
-   AWS Account (for S3 Bucket)
-   Google Cloud Account (for Gemini API)

---

## Part 1: Backend Deployment (Render)

We will use **Render** because it has native support for Python and is easy to set up.

### 1. Prepare for Deployment
Ensure your `backend/requirements.txt` is up to date.
```bash
cd backend
pip freeze > requirements.txt
```

### 2. Create a Web Service on Render
1.  Log in to [Render Dashboard](https://dashboard.render.com/).
2.  Click **New +** -> **Web Service**.
3.  Connect your GitHub repository.
4.  Configure the service:
    *   **Name**: `lumina-backend`
    *   **Root Directory**: `backend`
    *   **Runtime**: `Python 3`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn src.main:app --host 0.0.0.0 --port 10000`
5.  **Environment Variables**:
    Add the following secrets from your local `.env` file:
    *   `GOOGLE_API_KEY`: (Your Gemini Key)
    *   `AWS_ACCESS_KEY_ID`: (Your AWS Key)
    *   `AWS_SECRET_ACCESS_KEY`: (Your AWS Secret)
    *   `AWS_REGION`: (e.g., `us-east-1`)
    *   `S3_BUCKET_NAME`: (Your Bucket Name)
    *   `PYTHON_VERSION`: `3.9.0` (Optional, to match local)

6.  Click **Create Web Service**.

### 3. Get Backend URL
Once deployed, Render will give you a URL (e.g., `https://lumina-backend.onrender.com`). **Copy this URL.**

---

## Part 2: Frontend Deployment (Vercel)

We will use **Vercel** as it is optimized for Vite/React apps.

### 1. Update Frontend Configuration
You need to tell the frontend where the backend lives.
1.  Open `frontend/.env.production` (create if it doesn't exist).
2.  Add:
    ```env
    VITE_API_URL=https://lumina-backend.onrender.com
    ```
    *(Replace with your actual Render URL)*

### 2. Deploy to Vercel
1.  Log in to [Vercel Dashboard](https://vercel.com/).
2.  Click **Add New...** -> **Project**.
3.  Import your GitHub repository.
4.  Configure the project:
    *   **Framework Preset**: Vite
    *   **Root Directory**: `frontend` (Click "Edit" to select the `frontend` folder)
    *   **Build Command**: `npm run build`
    *   **Output Directory**: `dist`
5.  **Environment Variables**:
    *   Add `VITE_API_URL` with your Render Backend URL.
6.  Click **Deploy**.

---

## Part 3: Final Configuration

### 1. Update Backend CORS
Now that the frontend has a domain (e.g., `https://lumina-frontend.vercel.app`), you need to allow it in the backend.

1.  Go back to **Render Dashboard**.
2.  Go to **Environment Variables**.
3.  Add a new variable:
    *   `FRONTEND_URL`: `https://lumina-frontend.vercel.app`
4.  **Update Code**: You might need to update `backend/src/main.py` to use this variable for CORS:

    ```python
    # backend/src/main.py
    origins = [
        "http://localhost:5173",
        os.getenv("FRONTEND_URL") # Add this to your allowed origins
    ]
    ```

### 2. Redeploy Backend
Trigger a manual deploy on Render to apply the CORS changes.

---

## Troubleshooting

*   **Backend 500 Errors**: Check the "Logs" tab in Render. It usually indicates missing environment variables.
*   **CORS Errors**: Ensure the Vercel domain is exactly matched in the Backend's allowed origins (no trailing slash).
*   **Build Fails**: Ensure `requirements.txt` (Backend) and `package.json` (Frontend) are committed and in the root of their respective folders.
