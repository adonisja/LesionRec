# Interview Preparation Guide - LesionRec Project

This document captures the "Senior Engineer" interview questions, answers, and architectural concepts discussed during the development of the LesionRec project.

## 1. System Design: File Uploads

**Question:**
"We need to allow users to upload high-resolution images to S3. Should we use **Proxy Upload** (Client -> Server -> S3) or **Presigned URLs** (Client -> S3)?"

**The Answer:**
*   **Choice:** Proxy Upload (for this specific use case).
*   **Reasoning:** We need to perform server-side validation (face detection, acne check) *before* accepting the file. Storing invalid data in S3 wastes money and requires complex cleanup logic.
*   **Senior Nuance:** "I chose Proxy Upload for validation, but I would implement **streaming** to avoid loading the entire file into RAM, preventing Out-Of-Memory (OOM) crashes under load."

## 2. Security: CORS & Same-Origin Policy

**Question:**
"Browsers enforce the Same-Origin Policy. How do we allow communication between React (port 5173) and FastAPI (port 8000)? And how do we solve this in production?"

**The Answer:**
*   **Development:** Use **CORS (Cross-Origin Resource Sharing)** headers. The server explicitly tells the browser "I trust this origin."
*   **Production:** Use a **Reverse Proxy / Load Balancer** (e.g., Nginx, AWS ALB).
    *   `domain.com/` -> React
    *   `domain.com/api/` -> FastAPI
    *   **Benefit:** To the browser, everything is on the same origin (`domain.com`). This eliminates CORS issues entirely and improves performance (no pre-flight OPTIONS requests).

## 3. Mobile Experience: Resilient Uploads

**Question:**
"Mobile users have unstable connections. If a 5MB upload fails at 99%, standard HTTP fails completely. How do we fix this?"

**The Answer:**
*   **Concept:** **Multipart Uploads** (specifically AWS S3 Multipart Upload).
*   **Mechanism:** Break the file into small chunks (e.g., 5MB). Upload them individually. If one chunk fails, retry only that chunk.
*   **Benefit:** Allows for "Pause & Resume" functionality, critical for mobile networks.

## 4. Frontend Architecture: TypeScript

**Question:**
"TypeScript slows down development initially. How do you justify the cost to a stakeholder?"

**The Answer:**
*   **Concept:** **"Shift Left"**.
*   **Pitch:** "TypeScript shifts bug discovery from **Production** (expensive, reputation damaging) to **Development** (cheap, instant). The 'cost' of writing types is actually an insurance premium that pays out by preventing runtime crashes."

## 5. CSS Architecture: Tailwind vs. CSS-in-JS

**Question:**
"Tailwind clutters the HTML with classes. Why use it over Styled Components?"

**The Answer:**
*   **Defense:** **Consistency**. Tailwind forces all developers to use the same spacing scale (`m-4`), colors (`text-blue-500`), and typography. It prevents "pixel-pushing" discrepancies.
*   **Rebuttal:** "We solve clutter by extracting **Components** (`<PrimaryButton />`), not by extracting CSS classes. The complexity should live in the component logic, not hidden in a separate stylesheet."

## 6. Data Transfer: JSON vs. FormData

**Question:**
"Why can't we just send the image file inside a JSON object?"

**The Answer:**
*   **Technical Constraint:** JSON is text-based. Binary data (images) must be **Base64 encoded** to fit in JSON.
*   **The Cost:** Base64 encoding increases file size by **~33%**. This wastes bandwidth (expensive on mobile) and CPU cycles (encoding/decoding).
*   **Solution:** `FormData` sends raw binary bytes (multipart/form-data), which is the most efficient transport method.

## 7. Authentication: Build vs. Buy

**Question:**
"Why pay for AWS Cognito? Why not just create a `users` table in our database and store passwords there?"

**The Answer:**
*   **Security Risk:** Handling passwords requires **Salting**, **Hashing** (Argon2/bcrypt), and secure storage. One mistake leads to a data breach.
*   **Compliance:** Storing user data triggers **GDPR**, **CCPA**, and **SOC2** requirements.
*   **Maintenance:** You have to build "Forgot Password", "MFA", "Email Verification", and "Session Management" from scratch.
*   **Verdict:** "We buy Cognito to offload the *liability* and *maintenance* of identity management, allowing us to focus on our core business logic (Lesion Detection)."

## 8. Security: Token Storage (XSS vs. CSRF)

**Question:**
"Where should we store the JWT Access Token in the browser? LocalStorage or HttpOnly Cookies?"

**The Answer:**
*   **LocalStorage:**
    *   *Pros:* Easy to use with JS (`localStorage.getItem()`).
    *   *Cons:* **Vulnerable to XSS**. If a hacker injects a script, they can steal the token.
*   **HttpOnly Cookies:**
    *   *Pros:* **Immune to XSS**. JavaScript cannot read the cookie.
    *   *Cons:* Vulnerable to **CSRF** (Cross-Site Request Forgery) if not protected. Harder to implement (CORS issues).
*   **Senior Verdict:** "For maximum security, **HttpOnly Cookies** are superior. However, for Single Page Apps (SPAs), we often accept the risk of LocalStorage for convenience, provided we have strict Content Security Policies (CSP) to prevent XSS."

### 5. Dependency Management & "It Works on My Machine"

**Question:** A developer updates a core library (e.g., Tailwind v3 to v4) but forgets to update the syntax. It works for them but breaks for others. How do you prevent this?

**Answer:**
This is a failure of **Environment Standardization**.
1.  **Lockfiles (`package-lock.json` / `yarn.lock`):** These are the single source of truth. If a dev updates a package, the lockfile changes. If they don't commit the lockfile, teammates install different versions. *Always commit the lockfile.*
2.  **CI/CD Pipelines:** Your build server (GitHub Actions, Jenkins) acts as the "neutral referee." It runs `npm install` (clean) and `npm run build`. If the syntax is wrong, the build fails *before* the code can be merged.
3.  **Strict Versioning:** In `package.json`, using exact versions (removing `^` or `~`) can prevent accidental major upgrades, though it increases maintenance overhead.
4.  **Migration Scripts:** When upgrading major versions, look for "codemods" (e.g., `npx tailwindcss-upgrade`) that automatically rewrite your code to match the new syntax.

### 6. Authentication Security: Client Secrets in SPAs

**Question:** Why do we strictly forbid storing "Client Secrets" in a Single Page Application (React/Vue/Angular), and what specific attack does removing the secret prevent?

**Answer:**
1.  **Visibility:** SPAs run entirely in the user's browser. Any "secret" stored in the code (even in `.env` files bundled by Vite/Webpack) can be viewed by anyone using "View Source" or the Network tab. It is **impossible** to keep a secret in a public client.
2.  **The Attack:** If an attacker gets your Client Secret, they can impersonate your application. While they can't decrypt user passwords, they might be able to:
    *   Bypass rate limits.
    *   Spam your user pool with fake accounts.
    *   If the flow allows, potentially swap authorization codes for tokens on their own servers.
3.  **The Solution:** Use **Public Clients** (PKCE flow). Instead of a static secret, the app generates a temporary secret (Code Verifier) for each login attempt, hashes it (Code Challenge), and sends it to the server. This proves that the app requesting the token is the same one that initiated the login, without needing a long-term stored secret.

### 7. HealthTech Compliance: Third-Party AI APIs

**Question:** We are sending user images (PHI) to a public AI API (Google Gemini). What are the risks and how do we ensure HIPAA/GDPR compliance?

**Answer:**
1.  **The Risk:** Sending PHI to a standard public API often grants the provider rights to use that data for "service improvement" (training). This is a HIPAA violation.
2.  **Data Minimization (Anonymization):**
    *   **Scrub Metadata:** Remove EXIF data (GPS, device info) before sending.
    *   **Masking:** If the image contains a face or tattoo unrelated to the lesion, blur it (though difficult for facial acne).
    *   **Decoupling:** Never send the User ID or Email to the AI. Only send the raw pixel data.
3.  **Legal Framework (BAA):** For real production use, you must sign a **Business Associate Agreement (BAA)** with the cloud provider (Google Cloud / AWS). Enterprise versions of these APIs (e.g., Vertex AI on Google Cloud, not the public Gemini API) often support "Zero Data Retention" policies where they process the data in memory and delete it immediately without logging.
4.  **Architecture:**
    *   *Client -> Backend (Anonymize) -> AI -> Backend -> Database.*
    *   The AI never sees who the user is, only "Image X".

### 8. System Design: Image Formats & Storage Trade-offs

**Question:** In our privacy service, we convert all user uploads (likely JPEGs) to PNG format to strip metadata. What are the system-wide implications of this decision?

**Answer:**
This is a classic **"Purity vs. Performance"** trade-off.

1.  **Storage Costs (S3):**
    *   **JPEG:** Uses "lossy" compression. A high-quality photo might be **500KB**.
    *   **PNG:** Uses "lossless" compression. The same photo could easily be **5MB to 10MB**.
    *   *Impact:* Your S3 storage bill could increase by **10x**.

2.  **Network Latency (User Experience):**
    *   **Upload/Download:** Sending a 10MB PNG takes significantly longer than a 500KB JPEG, especially on mobile networks (3G/4G). This makes the app feel "slow."

3.  **Compute Overhead:**
    *   Encoding a PNG is computationally more expensive (CPU intensive) than saving a JPEG. This adds latency to your backend processing time.

4.  **The Verdict for LesionRec:**
    *   **For AI Analysis:** We *want* high fidelity. Compression artifacts (blurriness) in a JPEG could be mistaken for skin texture. PNG is safer for the *analysis* step.
    *   **For Archival:** Storing the PNG is expensive.
    *   **Optimization:** A middle ground is to strip metadata but save as **JPEG with 100% quality**. This removes EXIF but keeps file sizes reasonable (though still larger than standard JPEG).

**Example Scenario:**
> "If we have 10,000 users uploading 2 images a day:
> *   **JPEG (500KB):** 10GB/day.
> *   **PNG (5MB):** 100GB/day.
> Over a year, that's the difference between 3.6TB and 36TB of data to manage and pay for."

### 9. System Architecture: Ensemble AI Models

**Question:** We are combining Google Cloud Vision (for bounding boxes) and Gemini (for analysis). How does this impact latency, and how do we optimize it?

**Answer:**
1.  **The Latency Problem:** Calling API A (1.5s) and then API B (2.0s) sequentially results in a **3.5s** wait time for the user. This feels sluggish.
2.  **The Solution (Parallelism):** Since the two tasks are independent (Gemini doesn't need the bounding boxes to start its analysis), we should execute them **asynchronously in parallel**.
    *   *Python:* Use `asyncio.gather(task1, task2)`.
    *   *Result:* The total wait time becomes `max(1.5s, 2.0s) = 2.0s`. We save 1.5 seconds per request.
3.  **Cost Implication:** We are now paying for two API calls per upload. We must justify that the value (bounding boxes + analysis) exceeds the doubled cost.

### 10. Scalability: Thread Pools & Concurrency

**Question:** We are using `ThreadPoolExecutor` to run blocking API calls. If 1000 users upload images simultaneously, does the server spawn 2000 threads? What is the risk?

**Answer:**
1.  **Default Behavior:** By default, `loop.run_in_executor(None, ...)` uses a default `ThreadPoolExecutor`. In Python 3.8+, the limit is usually `min(32, os.cpu_count() + 4)`.
    *   *Result:* It will **NOT** spawn 2000 threads. It will spawn ~32 threads. The remaining 1968 tasks will sit in a queue waiting for a thread to become free.
    
2.  **The Risk (Latency vs. Crash):**
    *   **If Unbounded:** If we forced it to spawn 2000 threads, we would crash the server due to **Memory Exhaustion** (each thread needs stack memory) and **Context Switching** (the CPU spends all its time switching between threads instead of working).
    *   **With Default Limits:** The server won't crash, but **Latency** spikes. The 1000th user has to wait for ~60 previous batches to finish.

3.  **The Solution:**
    *   **True Async:** Use native async libraries (like `httpx` or `aiohttp`) instead of wrapping blocking code. This allows handling thousands of requests on a *single* thread without the overhead of OS threads.
    *   **Horizontal Scaling:** Spin up more server instances (Pods/Containers) behind a Load Balancer.

### 11. Error Handling: Partial Failures in Distributed Systems

**Question:** In a pipeline (Scrub -> Analyze -> Upload), if the **Upload to S3** fails but the **Analysis** succeeded, should we return the analysis to the user or fail the request?

**Answer:**
This is a question about **Data Consistency vs. User Experience**.

1.  **The Strict Engineering Approach (Recommended):** **Fail the Request.**
    *   *Reasoning:* If we return the analysis but fail to save the image, we have created a "Ghost Record." The user sees a result, but if they refresh the page or check their history, it's gone. This leads to confusion and support tickets ("I saw my diagnosis yesterday, where is it?").
    *   *Principle:* **Atomicity**. Treat the entire operation as a single transaction. Either it all succeeds, or it all fails.

2.  **The "Expensive Compute" Counter-Argument:**
    *   *Reasoning:* AI analysis costs money and time. Throwing away a successful result just because storage flickered seems wasteful.
    *   *Alternative:* Return the result with a warning (`saved: false`). However, this complicates the frontend logic significantly.

3.  **The Robust Solution:**
    *   Use a **Background Job**.
    *   1. Upload to S3 (quicker/safer).
    *   2. Return "Processing" to user.
    *   3. Worker pulls image, analyzes, saves result to DB.
    *   4. Frontend polls for result.
    *   *Benefit:* Decouples the failure modes.

---

## Session 2: Advanced Python Patterns, Algorithm Design & Image Processing

### 12. Code Organization: Class vs. Module Functions

**Question:** Should `ImageProcessor` be implemented as a **class with instance methods** or as a **module with pure functions**? What are the trade-offs?

**The Answer:**

**Use a class when you have:**
1. **Configuration State:** Settings that don't change between method calls (e.g., `box_color`, `inpaint_radius`)
2. **Dependency Injection:** Need to swap implementations for testing (e.g., mock inpainter)
3. **Logical Grouping:** Related operations that share common setup

**Key Insight:** *Classes are for configuration, functions are for transformations.*

**Example - ImageProcessor (class-based):**
```python
class ImageProcessor:
    def __init__(self, box_color=(255,0,0), thickness=3):
        self.box_color = box_color  # Configuration state
        self.thickness = thickness

    def draw_boxes(self, image_bytes, regions):
        # Stateless method - takes input, returns output
        # Uses self.box_color for config, but doesn't store request data
        pass
```

**Why NOT store request data in instance variables:**
- In FastAPI, requests are handled concurrently
- If you store `self.current_image`, multiple requests will overwrite each other (race condition)
- **Golden Rule:** Instance variables for configuration, method parameters for request data

**Comparison to your codebase:**
- `analysis.py` uses functions (Gemini API client is global, no config needed)
- `ProductRecommender` uses a class (loads CSVs once, reuses across requests)

---

### 13. functools.partial() vs Lambda Functions

**Question:** Why use `functools.partial()` instead of lambda for threading/async operations?

**The Answer:**

**With Lambda:**
```python
task = loop.run_in_executor(None, lambda: _run_gemini(image, mime))
# Traceback shows: File "<lambda>", line 1, in <lambda>
```

**With functools.partial():**
```python
task = loop.run_in_executor(None, functools.partial(_run_gemini, image, mime))
# Traceback shows: File "analysis.py", line 34, in _run_gemini
```

**Why it matters:**
1. **Better stack traces:** Shows actual function name instead of `<lambda>`
2. **Profiling:** Performance tools can identify bottlenecks by function name
3. **Debugging:** When you have multiple lambdas, they all look the same in error logs

**When to use each:**
- **partial():** Simple function calls, production code, when debugging matters
- **lambda:** Complex logic, one-liners, when clarity > debuggability

---

### 14. Algorithm Design: Greedy Knapsack for Product Bundling

**Question:** Walk through how the product bundling algorithm works and explain the time complexity.

**The Answer:**

**Problem:** Given a budget, select one product per category to maximize value.

**Algorithm (Greedy Approximation):**
```python
bundle = []
remaining_budget = budget_max

for category in ['cleanser', 'treatment', 'moisturizer']:
    # Filter affordable products
    affordable = products[products['price'] <= remaining_budget]

    # Calculate value score
    affordable['value_score'] = (
        rating * 0.6 +
        (reviews/max_reviews) * 0.3 -
        (price/max_price) * 0.1
    )

    # Pick best value product
    best = affordable['value_score'].idxmax()
    bundle.append(best)
    remaining_budget -= best['price']
```

**Time Complexity:**
- Per category: O(n log n) for sorting/value calculation
- Total: O(k * n log n) where k = number of categories

**Why "Greedy" is not optimal:**
- Picks best cleanser first, which limits budget for treatments
- A cheaper cleanser might enable a better treatment (higher total value)
- **Optimal solution requires Dynamic Programming** (O(n * budget * categories))

**When to use Greedy vs DP:**
| Factor | Greedy | Dynamic Programming |
|--------|--------|---------------------|
| Speed | Fast (ms) | Slow (seconds for large budgets) |
| Optimality | Approximate | Guaranteed optimal |
| Use Case | Product recommendations, real-time UX | Investment portfolios, critical optimization |

**For LesionRec:** Greedy is correct because:
1. User experience requires fast response (< 500ms)
2. "Good enough" recommendations are acceptable
3. The optimal bundle might save only $2-3 (not worth complexity)

---

### 15. pandas Performance: nlargest() vs idxmax()

**Question:** What's the time complexity of finding the single best product? Is there a more efficient method than `nlargest(1)`?

**The Answer:**

**Current approach:**
```python
best_product = affordable.nlargest(1, 'value_score').iloc[0]  # O(n) + overhead
```

**Better approach:**
```python
best_product_idx = affordable['value_score'].idxmax()  # O(n)
best_product = affordable.loc[best_product_idx]
```

**Why idxmax() is better:**
- `nlargest(k)` uses a heap (designed for top-K items)
- `idxmax()` is a simple linear scan (designed for single max)
- Less DataFrame copying overhead
- Clearer intent ("find max" vs "get top 1 sorted by")

**When to use each:**
| Method | Use Case | Example |
|--------|----------|---------|
| `.max()` | Need the value only | `max_price = df['price'].max()` |
| `.idxmax()` | Need the row with max value | `best_product_idx = df['value_score'].idxmax()` |
| `.nlargest(k)` | Need top K items (K > 1) | `top_5 = df.nlargest(5, 'rating')` |

---

### 16. Magic Numbers and Code Maintainability

**Question:** Why extract hard-coded values into constants? Isn't it just more code?

**The Answer:**

**Bad (Magic Numbers):**
```python
affordable['value_score'] = (
    affordable['rating'] * 0.6 +  # Why 0.6?
    (affordable['reviews'] / max_reviews) * 0.3 -  # Why 0.3?
    (affordable['price'] / max_price) * 0.1  # Why 0.1?
)
```

**Good (Named Constants):**
```python
class ProductRecommender:
    RATING_PRIORITY_WEIGHTS = {
        'rating': 0.6,
        'reviews': 0.3,
        'price': 0.1
    }

    PRICE_PRIORITY_WEIGHTS = {
        'rating': 0.4,
        'reviews': 0.2,
        'price': 0.4
    }

    def create_bundle(self, prioritize_rating=True):
        weights = (self.RATING_PRIORITY_WEIGHTS if prioritize_rating
                   else self.PRICE_PRIORITY_WEIGHTS)

        affordable['value_score'] = (
            rating * weights['rating'] +
            reviews_normalized * weights['reviews'] -
            price_normalized * weights['price']
        )
```

**Benefits:**
1. **Self-documenting:** Clear what the numbers represent
2. **Single source of truth:** Change in one place, not scattered across code
3. **Configurable:** Easy to make them constructor parameters later
4. **A/B testing:** Swap weight profiles to test different strategies

---

### 17. Error Handling: Specific vs Generic Exceptions

**Question:** In `main.py`, why catch specific exceptions (`KeyError`, `ValueError`) instead of just one generic `Exception`?

**The Answer:**

**Bad (Generic Catch-All):**
```python
try:
    # Upload, analyze, recommend
    pass
except Exception as e:
    raise HTTPException(status_code=500, detail="Server error")
```
**Problem:** User gets "500 Server Error" whether they:
- Uploaded invalid file format (400 Bad Request)
- S3 is down (503 Service Unavailable)
- Hit rate limit (429 Too Many Requests)

**Good (Granular Exception Handling):**
```python
try:
    # Upload, analyze, recommend
    pass
except AttributeError as e:  # Recommender method missing
    raise HTTPException(status_code=503, detail="Recommendation service error")
except KeyError as e:  # Missing analysis data
    raise HTTPException(status_code=500, detail="Analysis incomplete")
except requests.exceptions.RequestException as e:  # S3 upload failed
    raise HTTPException(status_code=500, detail="Storage service unavailable")
except ValueError as e:  # Invalid image data
    raise HTTPException(status_code=400, detail="Invalid image")
except Exception as e:  # Unexpected errors
    raise HTTPException(status_code=500, detail="Internal server error")
```

**Why this matters:**
1. **User Experience:** Correct HTTP status codes help frontend show appropriate messages
2. **Debugging:** Specific errors in logs help identify root cause faster
3. **Monitoring:** Alert on 5xx (your fault) vs 4xx (user fault) differently
4. **SLAs:** 4xx errors don't count against uptime targets

---

### 18. Image Processing: Normalized vs Pixel Coordinates

**Question:** Why use normalized coordinates (0.0-1.0) instead of pixel coordinates for bounding boxes?

**The Answer:**

**Normalized Coordinates:**
```json
{
  "x_min": 0.25,  // 25% from left edge
  "y_min": 0.30,  // 30% from top
  "x_max": 0.40,  // 40% from left
  "y_max": 0.50   // 50% from top
}
```

**Pixel Coordinates:**
```json
{
  "x_min": 200,  // 200 pixels from left
  "y_min": 150,  // 150 pixels from top
  "x_max": 320,  // 320 pixels from left
  "y_max": 250   // 250 pixels from top
}
```

**Pros of Normalized:**
✅ **Resolution-independent:** Works with 800px or 4000px images
✅ **AI-friendly:** Gemini doesn't need to know image dimensions
✅ **Frontend flexibility:** Easy to scale for different screen sizes
✅ **Industry standard:** Cloud Vision API uses this format

**Cons of Normalized:**
❌ Need conversion to pixels for drawing (simple multiplication)

**Pros of Pixel:**
✅ Direct drawing (no conversion)

**Cons of Pixel:**
❌ Image size-dependent (coordinates only valid for specific resolution)
❌ Gemini must know image dimensions (adds complexity)
❌ Hallucination risk (AI might output x=5000 for a 1000px image)

**Verdict:** Normalized coordinates are the professional choice for APIs.

---

### 19. OpenCV vs PIL: When to Use Each

**Question:** We use PIL for drawing bounding boxes. Why not use OpenCV for everything?

**The Answer:**

**PIL (Pillow):**
- **Best for:** Simple image operations (resize, crop, rotate, draw shapes)
- **Color format:** RGB (matches web standards)
- **API:** Pythonic, easy to learn
- **Performance:** Good for single images

**OpenCV (cv2):**
- **Best for:** Computer vision (object detection, tracking, inpainting)
- **Color format:** BGR (historical reasons from early cameras)
- **API:** C++ origins, less intuitive
- **Performance:** Optimized for video/batch processing

**For LesionRec:**
- **Bounding boxes (PIL):** Simple rectangles, RGB colors match our config
- **Inpainting (OpenCV):** Advanced algorithm, no PIL equivalent

**Color conversion pitfall:**
```python
# PIL uses RGB
box_color = (255, 0, 0)  # Red

# OpenCV uses BGR
image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
box_color_bgr = (0, 0, 255)  # Red in BGR!
```

---

### 20. Inpainting Approaches: Blur vs OpenCV vs AI

**Question:** We need to "remove" blemishes from images. What are the different approaches and trade-offs?

**The Answer:**

**Approach A: Simple Blur**
```python
blurred = image.filter(ImageFilter.GaussianBlur(radius=10))
result = Image.composite(blurred, image, mask)
```
- **Speed:** < 100ms
- **Quality:** Looks obviously fake, loses texture
- **Use case:** Quick demos, placeholders

**Approach B: OpenCV Inpainting** ✅ Recommended for LesionRec
```python
result = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
```
- **Speed:** ~500ms
- **Quality:** Preserves skin texture, good enough for "what if" visualization
- **Algorithms:**
  - `INPAINT_TELEA`: Faster, good for small blemishes
  - `INPAINT_NS` (Navier-Stokes): Slower, better texture preservation
- **Use case:** Production apps needing good-enough quality without ML

**Approach C: AI Inpainting (Stable Diffusion, LaMa)**
- **Speed:** 2-5 seconds (requires GPU)
- **Quality:** Photo-realistic results
- **Complexity:** Model weights, dependencies, potential API costs
- **Use case:** High-end beauty apps, professional photo editing

**Decision for LesionRec:**
OpenCV because:
1. Users want a "what if" preview, not professional retouching
2. 500ms latency is acceptable
3. No ML dependencies or ongoing costs
4. Works reliably offline

---

### 21. Division by Zero: Edge Case Handling

**Question:** In value score calculation, what happens if all products have 0 reviews?

**The Answer:**

**Vulnerable code:**
```python
affordable['value_score'] = (
    rating * 0.6 +
    (reviews / reviews.max()) * 0.3  # Division by zero if max=0!
)
```

**Defensive code:**
```python
max_reviews = affordable['reviews'].max()
reviews_score = ((affordable['reviews'] / max_reviews) * 0.3
                 if max_reviews > 0 else 0)

affordable['value_score'] = rating * 0.6 + reviews_score - price_score
```

**Why this matters:**
- Edge cases crash production systems
- **Senior engineers think about: "What if the dataset is empty? What if all values are zero?"**
- This is the difference between junior (happy path only) and senior (defensive programming)

**Other edge cases to consider:**
- Empty DataFrames (`if df.empty:`)
- Missing columns (`if 'rating' in df.columns:`)
- Null values (`df['price'].fillna(0)`)
- Out-of-bounds coordinates (`max(0.0, min(1.0, coord))`)

---

### 22. Type Hints: Consistency and Professional Code

**Question:** Why are type hints important beyond just IDE autocomplete?

**The Answer:**

**Without type hints:**
```python
def process(data, config):
    # Is data a list? dict? bytes?
    # Is config optional?
    pass
```

**With type hints:**
```python
def process(data: bytes, config: Optional[Dict[str, Any]] = None) -> bytes:
    # Crystal clear contract
    pass
```

**Benefits:**
1. **Self-documentation:** No need to read implementation to know what to pass
2. **Early error detection:** Tools like `mypy` catch type mismatches before runtime
3. **Refactoring safety:** Change a return type, find all callers that break
4. **Team onboarding:** New developers understand APIs instantly

**Production example from your code:**
```python
def _normalize_to_pixels(
    self,
    regions: List[Dict],  # Explicit: expects list of dicts
    image_width: int,     # Not float, not str
    image_height: int
) -> List[Dict]:          # Returns list of dicts
    pass
```

**Consistency matters:**
```python
# Inconsistent (confusing)
def __init__(
    self,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    box_thickness = 3,  # Missing type hint!
):

# Consistent (professional)
def __init__(
    self,
    box_color: Tuple[int, int, int] = (255, 0, 0),
    box_thickness: int = 3,
):
```

---

### 23. Logging vs Print Statements

**Question:** Why use `logger.info()` instead of `print()` in production code?

**The Answer:**

**Why print() is bad in production:**
```python
print("User uploaded image")  # Where does this go in production?
```
- No timestamp
- No log level (is this INFO, WARNING, ERROR?)
- Can't filter by severity
- Can't redirect to files/monitoring systems
- Doesn't integrate with CloudWatch/Datadog/Sentry

**Why logger is professional:**
```python
logger.info("User uploaded image")    # Informational
logger.warning("High latency: 3.2s")  # Concerning but not broken
logger.error("S3 upload failed", exc_info=True)  # Critical, include stacktrace
```

**Benefits:**
1. **Filtering:** In dev, show DEBUG. In prod, show WARNING+
2. **Structure:** Logs include timestamp, module, line number automatically
3. **Centralization:** All logs go to CloudWatch, searchable by severity
4. **Alerting:** Can trigger PagerDuty on ERROR logs

**Your code (correct usage):**
```python
logger.info(f"Extracted {len(regions)} blemish regions")  # Tracking
logger.warning(f"Invalid box width: {region}")  # Data quality issue
logger.error(f"Failed to parse JSON: {e}")  # Critical failure
```

---

### 24. Coordinate System Validation: Clamping

**Question:** Why clamp normalized coordinates to [0.0, 1.0]? Shouldn't they always be in range?

**The Answer:**

**The Reality:** AI models can hallucinate slightly out-of-bounds values due to floating point math.

**Example from Gemini:**
```json
{
  "x_min": -0.002,   // Slightly negative
  "y_max": 1.0001    // Slightly over 1.0
}
```

**Without clamping:**
```python
pixel_x = int(-0.002 * 1000)  # -2 pixels (invalid!)
# Drawing library crashes or draws outside image bounds
```

**With clamping:**
```python
x_min_norm = max(0.0, min(1.0, x_min_norm))  # Clamps to [0.0, 1.0]
pixel_x = int(0.0 * 1000)  # 0 pixels (safe)
```

**Why this is professional:**
- **Defensive programming:** Never trust external input (even from AI)
- **Graceful degradation:** Fix the error automatically instead of crashing
- **Logging:** Still log the warning so you know the AI is misbehaving

**Senior insight:** "External systems are unreliable. Your code should be the reliability layer."

---

### Summary of Key Learnings

1. **Architecture:** Classes for configuration, functions for transformations
2. **Performance:** Use `idxmax()` over `nlargest(1)`, run APIs in parallel with `asyncio.gather()`
3. **Code Quality:** Extract magic numbers, add type hints, use specific exceptions
4. **Algorithms:** Understand greedy vs optimal (DP), know when "good enough" is acceptable
5. **Libraries:** PIL for simple ops (RGB), OpenCV for computer vision (BGR)
6. **Production Mindset:** Defensive programming, logging, edge case handling, graceful degradation
