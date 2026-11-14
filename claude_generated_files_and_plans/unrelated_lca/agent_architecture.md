# Layered Cognitive Agent (LCA) Architecture
**Goal:** A dual-loop, memory-augmented agent capable of cross-app task generalization on a single GPU.

## 1. High-Level Structure
The system is divided into two distinct execution loops to solve the "Latency vs Intelligence" trade-off.

### **System 1: The Actor (Fast Loop)**
*   **Role:** Real-time interaction, "Muscle Memory".
*   **Latency:** < 100ms.
*   **Component:** `BehavioralCloningService` (Enhanced).
*   **Logic:** 
    *   Input: `PerceptionState` (ViT Embedding + OCR Keywords).
    *   Model: KNN (current) or Lightweight MLP (future).
    *   Action: Mouse/Keyboard events via `Robot`.
*   **Trigger:** Runs continuous loop when "Active".

### **System 2: The Learner (Slow Loop)**
*   **Role:** Reasoning, Error Correction, Memory Update.
*   **Latency:** 2-10 seconds.
*   **Component:** `AgenticAdvisor` / `LocalLlmService`.
*   **Trigger:**
    *   **Failure:** `VisionStateDetector` sees "Crash" or "Stuck".
    *   **Novelty:** `PerceptionService` sees unknown UI pattern.
    *   **Task Completion:** Goal achieved.
*   **Action:**
    *   Analyzes Screenshot via `LocalLlmService` (VLM).
    *   Updates `GlobalAffordanceMemory` (e.g., "Red Gear = Settings").
    *   Refines `System 1` Policy.

---

## 2. Perception Layer (The "Eyes")
*   **Component:** `PerceptionService` (New).
*   **Inputs:** `ScreenWatcherService` (Raw Pixels).
*   **Pipeline:**
    1.  **Visual:** `PythonExecutionService.extractFeatures("vit")` -> 1024-d Vector.
    2.  **Structural:** `Tesseract` (OCR) -> List of text nodes (Words, Coordinates).
    3.  **Synthesis:** Combines Visual + Structural into a `UI_Graph` snapshot.

## 3. Memory Architecture
*   **Component:** `MemoryDistributorService` (New).
*   **Stores:**
    1.  **Global Affordance Memory:**
        *   **Content:** Visual Vectors -> Meaning (e.g., [Vector] -> "Save Button").
        *   **Implementation:** `PythonExecutionService` (Vector Store).
    2.  **App-Specific Memory:**
        *   **Content:** Overrides (e.g., "In Notepad++, [Vector] -> Paste").
        *   **Implementation:** JSON/Map in Java (`AppStateService`).
    3.  **Strategy Memory:**
        *   **Content:** Successful Action Sequences.
        *   **Implementation:** Linked List of `DemonstrationStep`.

---

## 4. Implementation Roadmap
### Phase 1: The Eyes (Perception)
*   Create `PerceptionService`.
*   Integrate `Tess4j` for OCR.
*   Wire `PythonExecutionService` for ViT Embeddings.

### Phase 2: The Brain (Memory)
*   Enhance `NeuralForge` to support structured "Affordance" storage (Vector + Label).
*   Implement "Memory Recall" in `BehavioralCloningService`.

### Phase 3: The Loops (Execution)
*   **Fast Loop:** `BehavioralCloningService` queries `PerceptionService` -> Queries `Memory` (KNN) -> Acts.
*   **Slow Loop:** If Confusion > Threshold, Pause -> Call `LocalLlmService` -> Update Memory -> Resume.
