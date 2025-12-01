# LesionRec (Lumina) 🔬✨

**AI-Powered Skin Analysis & Personalized Product Recommendation System**

[![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Gemini AI](https://img.shields.io/badge/AI-Gemini%202.0%20Flash-8E75B2.svg)](https://deepmind.google/technologies/gemini/)
[![License](https://img.shields.io/badge/License-Personal%20%26%20Educational-green.svg)](#license)

Lumina (formerly LesionRec) is a modern, full-stack application that leverages advanced AI to analyze skin conditions and generate personalized skincare routines. It combines a privacy-focused image processing pipeline with a sophisticated recommendation engine to offer users actionable insights and product bundles that fit their budget.

---

## 🚀 Project Status: MVP Complete

This project has successfully reached **Minimum Viable Product (MVP)** status. It features a fully functional frontend, a robust backend API, and integrated AI services for real-time analysis.

---

## 🏗️ Architecture & Tech Stack

The application is built using a modern decoupled architecture:

### **Frontend (Client-Side)**
- **Framework**: React (Vite) with TypeScript
- **Styling**: Tailwind CSS for responsive, modern UI
- **State Management**: React Hooks + LocalStorage for persistence
- **Key Components**:
  - `ImageUpload`: Handles camera capture and file uploads.
  - `RecommendedProducts`: Displays analysis results and product bundles.
  - `ProductRoutine`: Visualizes the step-by-step skincare routine.

### **Backend (Server-Side)**
- **Framework**: FastAPI (Python)
- **AI Services**:
  - **Google Gemini 2.0 Flash**: Primary engine for dermatological analysis (condition detection, severity assessment).
  - **Google Vision API**: Used for privacy scrubbing (face detection/blurring) before analysis.
- **Data Processing**: Pandas for product dataset manipulation.
- **Storage**:
  - **S3 (AWS)**: Secure storage for uploaded images (presigned URLs).
  - **Local Data**: Curated CSV datasets for skincare products.

---

## ✨ Key Features

1.  **Privacy-First Analysis**: All images are automatically scrubbed (faces blurred) to remove personally identifiable information before being stored or analyzed.
2.  **Multi-Modal Input**: Supports both file uploads and live camera capture.
3.  **AI-Driven Insights**: Detects conditions like Acne, Rosacea, Eczema, etc., and determines severity (Mild, Moderate, Severe).
4.  **Smart Budgeting**:
    - **Infinite Budget Mode**: Defaults to showing the absolute best products.
    - **Dynamic Bundling**: Users can set a specific budget (e.g., ), and the system recalculates the optimal "Bundle" (Cleanser + Treatment + Moisturizer) to fit the sum within that limit.
5.  **Dual-List Recommendations**:
    - **The Bundle**: A cohesive routine where the *total cost* fits your budget.
    - **Individual Picks**: A list of top-rated items where *each item* fits your budget.
6.  **Persistence**: Analysis results are saved locally, allowing page refreshes without re-uploading images.

---

## 📂 Project Structure

### **Root Directory**
- `backend/`: Python FastAPI server and logic.
- `frontend/`: React application.
- `archive/`: Legacy code (Streamlit, YOLO models) and documentation.
- `docs/`: Current project documentation.
- `start-dev.sh`: Script to launch both frontend and backend.

### **Backend Breakdown (`backend/src/`)**
- **`main.py`**: The entry point for the FastAPI server. Defines endpoints `/upload` and `/recommend`.
- **`services/`**:
  - `analysis.py`: Handles interaction with Google Gemini API.
  - `privacy.py`: Uses Google Vision API to detect and blur faces.
  - `product_recommender.py`: The core logic engine. Contains the "Knapsack-style" algorithm for bundling and filtering logic.
  - `product_data_cleaner.py`: Utilities for cleaning and loading CSV data.
  - `image_processor.py`: Helper functions for image manipulation.
- **`data/`**: Contains the CSV files for different skin conditions (e.g., `acne_products.csv`, `rosacea_products.csv`).

### **Frontend Breakdown (`frontend/src/`)**
- **`App.tsx`**: Main application controller. Handles routing between Upload and Results views.
- **`components/`**:
  - `ImageUpload.tsx`: Manages file selection, camera streaming, and API upload calls.
  - `RecommendedProducts.tsx`: The results dashboard. Manages the budget state and displays the bundle/list.
  - `ProductRoutine.tsx`: Renders the "Bundle" as a visual step-by-step card grid.

---

## 🔄 Data Flow Guide

1.  **User Action**: User uploads an image or captures a photo via `ImageUpload.tsx`.
2.  **Frontend**: Sends `POST /upload` request with the image file to the Backend.
3.  **Backend (Privacy)**: `privacy.py` detects faces and blurs them.
4.  **Backend (Storage)**: Uploads the scrubbed image to AWS S3.
5.  **Backend (AI Analysis)**: `analysis.py` sends the scrubbed image to Gemini 2.0 Flash.
    - *Prompt*: "Analyze this skin image for conditions..."
    - *Response*: JSON containing `condition` (e.g., "acne"), `severity`, and `characterization`.
6.  **Backend (Recommendation)**: `product_recommender.py` takes the analysis:
    - Loads the relevant product CSV (e.g., `acne_products.csv`).
    - **Bundle Logic**: Selects a Cleanser, Treatment, and Moisturizer such that `Sum(Prices) <= Budget`. (Defaults to Infinite if no budget).
    - **List Logic**: Selects top-rated items where `Item_Price <= Budget`.
7.  **Response**: Backend returns the Analysis + Bundle + Recommendations to Frontend.
8.  **Frontend**: `App.tsx` saves data to `localStorage` and switches to `RecommendedProducts.tsx` view.
9.  **User Interaction**: User enters a new budget (e.g., ).
10. **Update**: Frontend calls `POST /recommend` with the *existing* analysis text and *new* budget. Backend recalculates and returns the new bundle.

---

## 🛠️ Getting Started

### Prerequisites
- Node.js (v16+)
- Python (v3.9+)
- Google Cloud Credentials (for Vision & Gemini)
- AWS Credentials (for S3)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/adonisja/LesionRec.git
    cd LesionRec
    ```

2.  **Environment Setup**:
    Create a `.env` file in the root (or `backend/`) with the following:
    ```env
    GOOGLE_APPLICATION_CREDENTIALS="path/to/your/google-creds.json"
    GEMINI_API_KEY="your_gemini_key"
    AWS_ACCESS_KEY_ID="your_aws_key"
    AWS_SECRET_ACCESS_KEY="your_aws_secret"
    AWS_REGION="us-east-1"
    S3_BUCKET_NAME="your-bucket-name"
    ```

3.  **Run the Application**:
    We have provided a convenience script to start both servers:
    ```bash
    ./start-dev.sh
    ```
    *This will start the Backend on `http://localhost:8000` and the Frontend on `http://localhost:5173`.*

---

## 📝 License

**Personal & Educational Use License**

Copyright (c) 2025 Akkeem

This project is designed to be a learning resource and is available for **personal and educational use only**.

**✅ You are free to:**
*   **Download & Run**: Install and run the application locally on your machine.
*   **Study & Learn**: Review the source code to understand how the AI, Backend, and Frontend components work together.
*   **Modify**: Experiment with the code for your own personal learning and hobby projects.

**❌ You may NOT:**
*   **Commercial Use**: Use this source code, in whole or in part, for any commercial purpose, business, or revenue-generating activity.
*   **Redistribute for Profit**: Sell, license, or monetize this code or any derivative works based on it.

*If you wish to use this software for commercial purposes, please contact the author for permission.*

---

## ⚠️ Medical Disclaimer

**This application is for educational and informational purposes only.**

*   **Not Medical Advice**: The analysis, insights, and product recommendations provided by Lumina are generated by Artificial Intelligence and are **not** a substitute for professional medical advice, diagnosis, or treatment.
*   **Consult a Professional**: Always seek the advice of a physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read on this application.
*   **No Doctor-Patient Relationship**: Use of this application does not create a doctor-patient relationship.

**General Disclaimer**

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
