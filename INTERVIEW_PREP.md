# Interview Preparation Guide

## 1. Text-to-Product Mapping (Fuzzy Matching)

**Scenario:** You implement a fuzzy matching logic where the AI recommends "gentle soap", but your catalog only has "Dove Men+Care". The fuzzy matcher might return nothing, or worse, it might match "Dove Chocolate" if you sold snacks.

**Question:** *How would you improve the reliability of this "Text-to-Product" mapping to ensure users don't see irrelevant products or zero results when valid alternatives exist?*

**Answer:**
To improve reliability beyond simple string matching, I would implement a two-tiered approach:

1.  **Semantic Search (Vector Embeddings):**
    *   Instead of matching keywords ("soap" == "soap"), we match *meaning*.
    *   We convert our product catalog into vector embeddings (using models like OpenAI's `text-embedding-3-small` or Google's `embedding-001`).
    *   When the user asks for "gentle soap", we convert that query into a vector and find the nearest neighbors in our product vector space. This would correctly identify "Dove Men+Care" as a semantic match for "gentle soap" even if the words don't overlap perfectly.

2.  **AI Function Calling (Tool Use):**
    *   Instead of asking the LLM to output a list of strings (which is prone to hallucination), we give the LLM a "tool" definition.
    *   We define a function `search_products(category: str, skin_type: str, benefits: List[str])`.
    *   The LLM then outputs a structured function call like `search_products(category="cleanser", benefits=["gentle", "hydrating"])`.
    *   We execute this query against our database filters. This guarantees that the results are valid products from our inventory, eliminating hallucinations.

---

## 2. System Design & Latency

**Scenario:** Your "Ensemble Analysis" (Gemini + Google Vision) takes about 4-6 seconds to complete. As you scale to 10,000 concurrent users, this synchronous blocking call will exhaust your server threads and cause timeouts.

**Question:** *How would you re-architect this system to handle high scale without degrading the user experience?*

**Answer:**
I would move from a **Synchronous (Blocking)** architecture to an **Asynchronous (Event-Driven)** architecture.

1.  **Queue-Based Processing:**
    *   When the user uploads an image, the API immediately returns a `202 Accepted` status and a `job_id`. It does *not* wait for the analysis.
    *   The image and metadata are pushed to a message queue (like AWS SQS, RabbitMQ, or Redis).

2.  **Worker Pool:**
    *   A separate fleet of "Worker" services pulls jobs from the queue. These workers handle the heavy lifting (calling Gemini/Vision APIs).
    *   This allows us to scale the workers independently of the web server. If traffic spikes, the queue just gets longer, but the web server remains responsive.

3.  **Real-Time Updates (WebSockets/Polling):**
    *   The frontend polls an endpoint `/api/jobs/{job_id}` to check status.
    *   Alternatively, we use WebSockets to push the result to the client as soon as the worker finishes.

---

## 3. Handling AI Hallucinations

**Scenario:** A user uploads a picture of a serious skin infection. The AI confidently diagnoses it as "mild acne" and recommends over-the-counter cream. This is a dangerous medical error.

**Question:** *How do you implement safety guardrails to prevent the AI from giving dangerous medical advice?*

**Answer:**
We need a "Defense in Depth" strategy:

1.  **System Prompt Engineering:**
    *   Explicitly instruct the AI: "You are not a doctor. Do not provide definitive diagnoses. If a condition looks severe, infected, or ambiguous, you MUST recommend seeing a professional."

2.  **Confidence Thresholds:**
    *   If the AI's confidence score (or the Google Vision confidence score) is below a certain threshold (e.g., 85%), the UI should display a "Low Confidence" warning.

3.  **"Human-in-the-Loop" (HITL) or Heuristics:**
    *   We can use a secondary, smaller model (or a classification layer) specifically trained to detect "medical emergencies" (e.g., open wounds, severe rashes). If this model flags the image, we block the standard recommendation flow and show a "Seek Medical Attention" banner immediately.

4.  **Disclaimer & UI Friction:**
    *   Always display a prominent medical disclaimer.
    *   For severe classifications, require the user to click "I understand this is not a medical diagnosis" before revealing the results.
