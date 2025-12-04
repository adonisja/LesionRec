# Lumina Demo Guide: Talking Points & Q&A

## Part 1: Core Talking Points

### 1. Introduction (The "Hook")
*   "Lumina is an AI-powered dermatological assistant designed to bridge the gap between skin analysis and actionable product routines."
*   "Unlike generic skincare quizzes, Lumina uses **computer vision** to analyze your actual skin condition and builds a custom regimen based on real-time product data."

### 2. The Architecture (The "How")
*   **Tech Stack**: "We built this using a decoupled architecture: A **React/Vite** frontend for a responsive experience, and a **FastAPI** backend for high-performance async processing."
*   **The AI Ensemble**: "We don't rely on a single model. We use an ensemble approach:
    *   **Google Vision API** for privacy (face detection).
    *   **Gemini 2.5 Pro** for the dermatological reasoning and spatial awareness.
    *   **Custom Python Algorithms** for the mathematical product optimization."

### 3. Key Feature: Privacy-First Pipeline
*   *Demo Action: Upload an image.*
*   "Before this image is ever analyzed for acne, it goes through our **Privacy Pipeline**. We perform **in-memory EXIF scrubbing** to remove GPS metadata and apply **automatic face blurring** so the AI analyzes the skin, not the identity."

### 4. Key Feature: The "Knapsack" Recommendation Engine
*   *Demo Action: Show the 'Recommended Products' screen.*
*   "This isn't just a random list of products. We implemented a variation of the **Knapsack Algorithm**."
*   "The system takes your budget (e.g., $50) and mathematically calculates the combination of Cleanser, Toner, and Treatment that maximizes 'Dermatological Value' without exceeding your limit."
*   "We also use **Weighted Scoring**: A 4.8-star product with 10,000 reviews is valued higher than a 5.0-star product with only 2 reviews."

### 5. Key Feature: Dynamic Budgeting
*   *Demo Action: Change budget from "No Limit" to "$30" and hit Update.*
*   "Notice how the routine adapts instantly. The AI swaps out the premium serum for a high-value budget alternative, ensuring you still get a complete routine regardless of price point."

## Part 2: Advanced Technical Talking Points

### 1. The "Resilient Parsing" System
*   "One of the biggest challenges with LLMs is getting consistent, structured output. We solved this with a **Dual-Layer Parsing Strategy**."
*   "First, we enforce a strict **JSON Schema** at the prompt level. If the model slips and returns unstructured text (or invalid JSON), our system automatically falls back to a **Regex-based Keyword Extractor**. This ensures the user *always* gets a recommendation, even if the LLM hiccups."

### 2. Model Upgrade: Gemini 2.5 Pro
*   "We are running on **Gemini 2.5 Pro**. We chose this specific model because of its superior **spatial awareness**."
*   "We prompt the model to return `blemish_regions` with normalized `(x, y)` coordinates. 2.5 Pro is significantly better at accurately localizing these features compared to smaller models, allowing us to map lesions precisely on the face."

### 3. The "Curated Dataset" Approach
*   "Instead of scraping random products from Amazon in real-time (which is slow and risky), we use a **Curated Data Pipeline**."
*   "Our `ProductDataCleaner` class pre-processes raw product data, normalizing prices and categorizing items (e.g., detecting that 'CeraVe PM' is a 'Moisturizer'). This means our recommendation engine runs in **milliseconds** because it's querying a structured, in-memory dataset rather than making external network calls."

### 4. Asynchronous "Ensemble" Execution
*   "To minimize latency, we use Python's `asyncio` library to run our AI models in parallel."
*   "We dispatch the **Google Vision** task (for object detection) and the **Gemini** task (for dermatological reasoning) simultaneously. We then use `await asyncio.gather()` to merge the results. This cuts our total processing time by roughly 40%."

## Part 3: Anticipated Q&A

### General & Ethical Questions

**Q1: "How do you prevent the AI from hallucinating medical advice?"**
> **Response:** "We use a two-layer safety approach. First, we prompt the model with a strict **JSON Schema** that forces it to output structured data (Acne Type, Severity) rather than free-text medical advice. Second, we include a hard-coded system prompt that forces the model to act as an 'educational assistant' and explicitly flag that it is not a doctor. The frontend also displays a mandatory Medical Disclaimer."

**Q2: "Do you store the user's photos? Is that safe?"**
> **Response:** "Images are stored in a private AWS S3 bucket solely for the duration of the analysis session. We strip all metadata (GPS, Device ID) *before* upload. In a production environment, we would configure a **Lifecycle Policy** on the S3 bucket to auto-delete images after 24 hours to ensure zero long-term data retention."

**Q3: "Why did you choose Gemini 2.5 Pro over GPT-4?"**
> **Response:** "Speed and multimodal capability. Gemini 2.5 Pro has a significantly lower latency for image processing, which is critical for the user experience. We need the analysis to happen in seconds, not minutes, to keep the user engaged."

**Q4: "How does the recommendation algorithm actually work?"**
> **Response:** "It's a value-optimization problem. We assign every product a 'Value Score' based on its rating, review count, and ingredient match. If you set a $50 budget, the algorithm iterates through thousands of combinations to find the set of products that gives the highest total 'Value Score' while keeping the sum of prices under $50."

**Q5: "What happens if the camera doesn't work on my phone?"**
> **Response:** "We implemented a robust fallback system. It attempts to access the camera stream, but if permissions are denied or the hardware is unavailable, it gracefully degrades to a standard file picker. We also handle mobile-specific issues like video mirroring (selfie mode) automatically."

### Technical "Deep Dive" Questions

**Q6: "What happens if the AI API goes down during a demo?"**
> **Response:** "We built a fail-safe in our backend. If the Gemini API throws an exception (timeout, rate limit), the system catches it and returns a 'Safe Fallback' JSON object. The frontend will display a polite 'Analysis Unavailable' message with general advice, rather than crashing with a white screen."

**Q7: "How do you handle the 'Cold Start' problem? (i.e., recommending products to a new user with no history)"**
> **Response:** "That's the beauty of Computer Vision. We don't *need* user history. The user's 'profile' is generated instantly from their image. The visual features (redness, pustules, dry patches) act as the immediate signal for our recommendation engine, solving the cold start problem completely."

**Q8: "Why do you normalize the ratings in your algorithm?"**
> **Response:** "We found that a 5.0-star product with 1 review isn't actually better than a 4.8-star product with 10,000 reviews. We normalize ratings to a 0-1 scale and apply a weighted score (0.6 for Rating, 0.3 for Review Count). This mathematically biases the system toward 'Trusted' products over 'Perfect' but unknown ones."

**Q9: "Is the product data hard-coded?"**
> **Response:** "No, it's data-driven. We load product datasets dynamically. This allows us to update our catalog (e.g., adding a new 'Summer Sunscreen' line) simply by updating the underlying CSV/Database, without changing a single line of the recommendation logic."

**Q10: "How do you ensure the bounding boxes match the face?"**
> **Response:** "We use **Normalized Coordinates** (0.0 to 1.0) rather than absolute pixels. This makes our data resolution-independent. Whether the user uploads a 4K selfie or a low-res webcam capture, the relative position of the blemish remains accurate."
