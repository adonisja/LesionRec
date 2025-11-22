# Dashboard Testing Guide

**Dashboard URL:** http://localhost:8501

---

## ✅ Test Checklist

### Test 1: Verify Dashboard Loads
- [ ] Dashboard opens in browser
- [ ] See "LesionRec" header
- [ ] Sidebar shows available models
- [ ] Upload area visible

**Expected:**
```
✅ YOLOv8
❌ YOLOv10 (still training)
❌ Gemini Vision (needs API key)
✅ Product Recommendations
```

---

### Test 2: Upload and Analyze with YOLOv8

**Steps:**
1. Click "Browse files" or drag and drop
2. Navigate to: `data/yolo_dataset/images/test/`
3. Select any image (try `C0018492-Acne.jpg`)
4. Make sure "YOLOv8" is selected in sidebar
5. Click "🔍 Analyze Skin" button

**Expected Results:**
- Original image displays
- Annotated image shows with bounding boxes
- Lesion count appears (comedones, papules, etc.)
- Confidence scores shown
- Product recommendations appear below

**If it works:** ✅ YOLOv8 detection is working!

**If you see errors:**
- Check console/terminal for error messages
- Copy error and share with me

---

### Test 3: Product Recommendations

**Should automatically show after analysis:**

Expected sections:
- 🧼 **CLEANSER** (with specific product name)
- 💊 **TREATMENT** (1-2 products)
- 💧 **MOISTURIZER**
- 📅 **DAILY ROUTINE** (morning/evening steps)
- ⏱️ **Timeline** (expected results timeframe)
- 💡 **Pro Tip**

**If it works:** ✅ Product system is working!

---

### Test 4: Try Different Images

Upload these test images one by one:

1. **`acne-open-comedo-3.jpg`** - Should detect comedones
2. **`Before.jpg`** - General acne
3. **`C0018492-Acne.jpg`** - Multiple lesions

**Check:**
- [ ] Each image analyzes successfully
- [ ] Detection counts vary by image
- [ ] Product recommendations change based on detected types

---

### Test 5: Gemini Integration (Optional - if you have API key)

**Setup:**
```bash
export GEMINI_API_KEY="your_key_here"
```

**Then restart dashboard:**
```bash
# Stop: Ctrl+C in terminal
# Restart:
streamlit run app/streamlit_app.py
```

**Expected after restart:**
- Sidebar shows ✅ Gemini Vision
- Can select "Gemini Vision" in model dropdown
- Upload image → analyze
- See natural language analysis in Gemini tab

---

## 🐛 Common Issues & Fixes

### Issue: "YOLOv8 model not available"
**Fix:**
```bash
# Check if model exists
ls -lh runs/detect/acne_yolov8_production/weights/best.pt

# If missing, model path might be wrong
# Check actual location:
find . -name "best.pt" -path "*/yolov8*"
```

### Issue: "ModuleNotFoundError"
**Fix:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: Upload button not working
**Fix:**
- Try smaller image (< 10MB)
- Use JPG/PNG format only
- Check file isn't corrupted

### Issue: Streamlit won't start
**Fix:**
```bash
# Kill existing processes
pkill -f streamlit

# Restart
streamlit run app/streamlit_app.py
```

### Issue: Products not showing
**Fix:**
```bash
# Check database exists
cat data/products.json | head -20

# If missing, let me know
```

---

## 📸 What Good Results Look Like

### YOLOv8 Detection:
```
Total Lesions: 8
🔴 Comedones: 2
🟠 Papules: 4
🟡 Pustules: 2
🔵 Nodules: 0
Avg Confidence: 87.3%
```

### Product Recommendations:
```
Severity: MODERATE
Dominant Type: PAPULES

🧼 CLEANSER:
• CeraVe Acne Foaming Cream Cleanser
  Ingredient: Benzoyl Peroxide 4%
  Price: $15-18

💊 TREATMENT:
• Differin Gel (Adapalene 0.1%)
  Why: Prevents clogged pores, reduces inflammation
```

---

## 🎯 Testing Complete When:

- [x] Dashboard loads without errors
- [x] Can upload images
- [x] YOLOv8 detects lesions
- [x] Bounding boxes appear on image
- [x] Product recommendations show
- [x] Can test multiple images
- [ ] Gemini integration works (optional until you get API key)
- [ ] YOLOv10 works (once training complete)

---

## 📝 Report Back

After testing, let me know:

1. **What works?** (e.g., "YOLOv8 detection works perfectly!")
2. **What doesn't work?** (e.g., "Upload button does nothing")
3. **Any error messages?** (copy the exact error)
4. **Screenshots?** (if possible, share what you see)

---

**Next Steps After Successful Test:**
1. Get Gemini API key for natural language analysis
2. Wait for YOLOv10 training to complete
3. Test with more diverse images
4. Prepare demo presentation

---

**Need help?** Just tell me what's happening and I'll help debug!
