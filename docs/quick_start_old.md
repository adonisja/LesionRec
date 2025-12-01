# 🚀 Quick Start Guide - LesionRec

## One-Command Startup

### macOS/Linux
```bash
./start-dev.sh
```

### Windows
```cmd
start-dev.bat
```

That's it! The script will:
- ✅ Check prerequisites (Python, Node.js, npm)
- ✅ Create virtual environment (if needed)
- ✅ Install dependencies
- ✅ Start backend on http://localhost:8000
- ✅ Start frontend on http://localhost:5173
- ✅ Show live logs

Press **Ctrl+C** to stop both servers.

---

## Manual Startup (Alternative)

### Terminal 1 - Backend
```bash
source .venv/bin/activate
cd backend
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2 - Frontend
```bash
cd frontend
npm run dev
```

---

## First-Time Setup

### 1. Prerequisites
- Python 3.9+ (`python3 --version`)
- Node.js 16+ (`node --version`)
- npm 8+ (`npm --version`)

### 2. Environment Variables
Create `.env` file in project root:
```bash
# Google AI
GOOGLE_API_KEY=your_gemini_api_key_here

# AWS S3
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name

# Google Cloud Vision (optional)
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
```

### 3. Install Dependencies

**Backend:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

---

## Testing Camera Capture

### Quick Test Flow:
1. Open http://localhost:5173
2. Click **"Use Camera to Capture Photo"**
3. Allow camera permissions
4. See live mirrored video preview
5. Click **"Capture Photo"**
6. Review captured image
7. Click **"Upload & Analyze"**
8. Wait 2-5 seconds for AI analysis
9. See results and product recommendations

### Detailed Testing:
See [TESTING_CHECKLIST.md](./TESTING_CHECKLIST.md) for comprehensive test cases.

---

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000 (backend)
lsof -i :8000
kill -9 <PID>

# Kill process on port 5173 (frontend)
lsof -i :5173
kill -9 <PID>
```

### Camera Not Working
- **Browser permissions:** Check browser settings → Site Permissions → Camera
- **HTTPS required:** Camera only works on localhost (HTTP) or HTTPS domains
- **Mobile iOS:** May need to enable camera in Settings → Safari → Camera

### Module Not Found
```bash
# Backend
source .venv/bin/activate
pip install -r backend/requirements.txt

# Frontend
cd frontend
npm install
```

### API Errors
- Check `.env` file has correct API keys
- Verify AWS credentials with: `aws s3 ls` (install AWS CLI)
- Test Gemini API: Visit http://localhost:8000/docs → Try `/upload` endpoint

---

## Useful Commands

### View Logs
```bash
# Backend logs
tail -f logs/backend.log

# Frontend logs
tail -f logs/frontend.log
```

### Check Server Status
```bash
# Backend health check
curl http://localhost:8000/

# Expected response:
# {"status":"healthy","recommender_available":true,"s3_available":true}
```

### Stop Servers
```bash
# If using start-dev.sh
Press Ctrl+C in terminal

# Manual kill
pkill -f "uvicorn src.main:app"
pkill -f "vite"
```

---

## Project Structure

```
LesionRec/
├── backend/
│   ├── src/
│   │   ├── main.py                    # FastAPI server
│   │   └── services/
│   │       ├── analysis.py            # AI ensemble (Gemini + Vision)
│   │       ├── image_processor.py     # Bounding boxes + inpainting
│   │       ├── product_recommender.py # Product matching
│   │       └── privacy.py             # Metadata scrubbing
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   └── components/
│   │       └── ImageUpload.tsx        # Camera capture + file upload
│   └── package.json
├── start-dev.sh                       # Startup script (macOS/Linux)
├── start-dev.bat                      # Startup script (Windows)
├── TESTING_CHECKLIST.md              # Comprehensive test cases
└── .env                              # Environment variables (create this)
```

---

## URLs

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:5173 | React UI |
| Backend | http://localhost:8000 | FastAPI server |
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| Health Check | http://localhost:8000/ | Server status |

---

## Next Steps

1. ✅ Run the startup script
2. ✅ Test camera capture feature
3. ✅ Test file upload feature
4. ✅ Check [TESTING_CHECKLIST.md](./TESTING_CHECKLIST.md)
5. ✅ Review backend logs for any errors
6. ✅ Test on mobile devices (optional)

---

## Support

- **Documentation:** See [INTERVIEW_PREP.md](./INTERVIEW_PREP.md) for technical deep-dive
- **API Endpoints:** http://localhost:8000/docs
- **Testing Guide:** [TESTING_CHECKLIST.md](./TESTING_CHECKLIST.md)

Happy testing! 🎉
