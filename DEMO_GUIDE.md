# Lumina Demo Guide & Q&A 🎤

This guide is designed to help you present **Lumina** effectively. It covers a step-by-step demo flow, key talking points, and a comprehensive list of anticipated questions to help you handle Q&A sessions with confidence.

---

## 🎬 Demo Walkthrough Script

### 1. Introduction (The "Why")
*   **Start**: "Skin health is complex, and finding the right products can be overwhelming and expensive. Lumina solves this by combining medical-grade AI analysis with personalized, budget-aware product recommendations."
*   **Visual**: Show the Landing Page/Dashboard.

### 2. The Dashboard
*   **Action**: Log in (if auth is enabled) or land on the Dashboard.
*   **Talk Track**: "Here is our central hub. We can start a new analysis, view our history, or jump straight to our AI assistant."

### 3. Image Upload & Privacy
*   **Action**: Click "Start Analysis". Upload a sample image (or use the camera).
*   **Talk Track**: "Privacy is paramount. Before this image is even analyzed, our system uses Google Vision API to detect and blur faces, ensuring no personally identifiable biometric data is stored or processed for the diagnosis."

### 4. AI Analysis (The "Magic")
*   **Action**: Wait for the loading spinner.
*   **Talk Track**: "We are now running an ensemble analysis. We use Google's Gemini 2.0 Flash for its multimodal reasoning capabilities to detect conditions like acne, rosacea, or eczema, and assess their severity."

### 5. Results Page
*   **Action**: Show the Analysis Results page. Point out the bounding boxes (if applicable) and the severity badge.
*   **Talk Track**: "The AI has identified [Condition] with [Severity]. It provides a detailed observation and identifies specific blemish regions."

### 6. Product Recommendations (The "Value")
*   **Action**: Click "View Products".
*   **Talk Track**: "This is where Lumina shines. Instead of just a random list, we generate a **Personalized Bundle**."
*   **Demo Feature**: Enter a budget (e.g., $50) and click "Update Bundle".
*   **Talk Track**: "Watch how the system recalculates. It performs a 'Knapsack' optimization to find the best combination of Cleanser, Treatment, and Moisturizer that fits *collectively* within your $50 limit."

### 7. Catalog & Sorting
*   **Action**: Scroll down to "All Matching Products". Use the Sort Dropdown (e.g., "Price: Low to High").
*   **Talk Track**: "For power users, we offer the full catalog. You can sort by price, rating, or popularity to find exactly what you need."

### 8. AI Chatbot
*   **Action**: Navigate to the "AI Assistant" tab. Ask: *"What is the best way to apply retinol?"*
*   **Talk Track**: "For questions that go beyond the image, our context-aware chatbot provides dermatological advice, remembering your specific analysis context."

---

## ❓ Anticipated Q&A (25 Questions)

### 🟢 General & Product

**Q1: Who is the target audience for Lumina?**
*   **A:** Anyone struggling with skin issues who finds the dermatological system expensive or inaccessible, and wants data-driven product recommendations.

**Q2: Is this intended to replace a dermatologist?**
*   **A:** No. Lumina is a triage and support tool. It provides "pre-diagnosis" insights and over-the-counter recommendations. For severe cases, we always recommend seeing a professional.

**Q3: How does the "Bundle" logic work?**
*   **A:** Unlike standard e-commerce sites that filter items individually, our "Bundle" logic looks at the *sum* of the products. It tries to fit a complete routine (Cleanser + Treatment + Moisturizer) into your total budget.

**Q4: Can I buy the products directly on the app?**
*   **A:** Currently, we provide affiliate links to major retailers like Amazon. Direct purchasing is on our roadmap.

**Q5: What happens if I don't set a budget?**
*   **A:** The system defaults to "Infinite Budget" mode, prioritizing the highest-rated and most effective products regardless of cost.

### 🟡 Technical & Architecture

**Q6: What is the tech stack?**
*   **A:** The frontend is **React (Vite) with TypeScript**. The backend is **FastAPI (Python)**. We use **AWS S3** for storage and **Google Cloud (Gemini + Vision)** for AI.

**Q7: Why did you choose FastAPI over Django or Flask?**
*   **A:** FastAPI offers superior performance (async support), automatic API documentation (Swagger UI), and strict type validation with Pydantic, which reduces runtime errors.

**Q8: How do you handle the connection between Frontend and Backend?**
*   **A:** We use RESTful API endpoints. The frontend makes asynchronous fetch requests to the backend, which processes the data and returns JSON responses.

**Q9: How is the application hosted?**
*   **A:** Currently, it runs in a local development environment. For production, we would containerize it with Docker and deploy to AWS (ECS or App Runner) or Vercel/Render.

**Q10: I noticed a "Compatibility Patch" in the code. What is that?**
*   **A:** We encountered an issue with a dependency requiring Python 3.10 features while running on Python 3.9. We implemented a shim to backport `importlib.metadata` functionality to ensure stability across environments.

### 🔵 AI & Machine Learning

**Q11: Which AI models are you using?**
*   **A:** We use a multi-model approach. **Google Vision API** handles object localization and face detection. **Gemini 2.0 Flash** handles the complex dermatological reasoning and text generation.

**Q12: Why Gemini instead of GPT-4?**
*   **A:** Gemini 2.0 Flash is multimodal-native, meaning it processes images and text simultaneously with high speed and lower latency, which is crucial for a responsive user experience.

**Q13: How accurate is the diagnosis?**
*   **A:** While we don't claim medical diagnostic accuracy, our testing shows high concordance with common dermatological classifications for acne, rosacea, and eczema.

**Q14: Does the model hallucinate?**
*   **A:** All LLMs can hallucinate. We mitigate this by using strict system prompts, constraining the output to specific JSON schemas, and validating the response against known skin conditions before displaying it.

**Q15: How do you handle different skin tones?**
*   **A:** This is a critical focus. We prompt the model to consider Fitzpatrick skin types and use diverse datasets to ensure the recommendations are inclusive and effective for all skin tones.

### 🟣 Privacy & Security

**Q16: Do you store my photos?**
*   **A:** We store the *scrubbed* (anonymized) version in a private AWS S3 bucket to allow you to view your history. We do not store the raw, unblurred images permanently.

**Q17: How do you ensure privacy?**
*   **A:** We use an "Edge-to-Cloud" privacy strategy. Before the image is analyzed for medical insights, it passes through a privacy filter that detects and blurs faces.

**Q18: Is the data encrypted?**
*   **A:** Yes, data is encrypted in transit (HTTPS) and at rest (AWS S3 Server-Side Encryption).

**Q19: Can other users see my results?**
*   **A:** No. Access is restricted via authentication (AWS Cognito/Amplify). You can only access data associated with your unique User ID.

**Q20: What if the AI leaks my data?**
*   **A:** We use enterprise-grade API endpoints that do not train on user data by default, ensuring your inputs remain private to your session.

### ⚪️ Future Roadmap

**Q21: What's next for Lumina?**
*   **A:** We plan to implement "Progress Tracking," allowing users to upload photos over time to generate a timelapse of their skin's improvement.

**Q22: Will you add more conditions?**
*   **A:** Yes, we are expanding the dataset to cover hair and scalp conditions (e.g., alopecia, dandruff).

**Q23: Can you integrate with wearable devices?**
*   **A:** We are exploring integrations with smart mirrors and UV trackers to provide environmental context to the analysis.

**Q24: How will you monetize?**
*   **A:** Potential revenue streams include affiliate commissions from product sales, a premium subscription for advanced tracking, and partnerships with dermatologists.

**Q25: Are you planning a mobile app?**
*   **A:** The current React frontend is fully responsive (PWA-ready). A native React Native app is a logical next step for better camera integration.

---

## 🛠️ Troubleshooting During Demo

*   **If the upload fails:** Check the file size (limit is 10MB) and format (JPG/PNG). Ensure the backend server is running.
*   **If the analysis is slow:** This depends on the Gemini API latency. It usually takes 5-10 seconds. Keep the "Talk Track" going to fill the silence.
*   **If the products are empty:** Ensure the detected condition exists in our database. Try a generic "acne" image if a rare condition fails.
