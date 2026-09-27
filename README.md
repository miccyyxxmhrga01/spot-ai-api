# SPOT AI V1 — Visual Talent Tagging

This project is the first AI component for **SPOT**, the campus talent showcase app.
It analyzes an uploaded still image and returns **suggested visual talent tags** for the Android app.

It does **not** train SPOT's recommendation engine, collaboration matcher, feedback assistant, or caption generator. Those are separate systems with different data requirements.

## Why the V1 classes changed

V1 uses only visual classes that can reasonably be inferred from a still image:

- `guitar` → Music
- `piano` → Music
- `drums` → Music
- `painting` → Art
- `drawing` → Art
- `dancing` → Dance
- `other` → no supported visual talent detected

`singing` is intentionally excluded because a still photo of a microphone does not prove that someone is singing. Singing should later use **audio/video analysis**.

`photography` is intentionally excluded from this classifier because every input is already a photograph/image and the label overlaps other classes. A future photography-specific model can classify styles such as portrait, landscape, street, event, etc.

## Folder structure

```text
SPOT_AI/
├── dataset/
│   ├── train/
│   │   ├── dancing/
│   │   ├── drawing/
│   │   ├── drums/
│   │   ├── guitar/
│   │   ├── other/
│   │   ├── painting/
│   │   └── piano/
│   ├── validation/
│   │   └── same class folders...
│   └── test/
│       └── same class folders...
├── models/
├── reports/
├── src/
│   ├── config.py
│   ├── dataset_audit.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── api_server.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## 1. Collect the dataset

A practical target is roughly:

- **70%** training
- **15%** validation
- **15%** test

For a school project, 100 training images per class is a rough minimum. **300–500+ diverse training images per class is much better.**

Diversity matters more than collecting many nearly identical pictures. Vary:

- people
- backgrounds
- indoor/outdoor settings
- camera angle
- distance
- lighting
- device/image quality
- poses and performance situations

Do not place the same image in multiple splits. If several frames/photos came from the same person/session/video, keep that group in only one split when possible. This reduces data leakage.

### What goes in `other/`?

Use normal images that do not clearly show one of the six supported talents, for example:

- classrooms
- buildings
- food
- animals
- ordinary selfies
- books
- cars
- landscapes
- computers
- unrelated campus scenes

`other` teaches the model that not every uploaded image must be forced into a talent category.

## 2. Install

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3. Audit the dataset first

```bash
python src/dataset_audit.py
```

The audit checks:

- required folders
- image counts
- class imbalance
- unreadable/corrupt images
- exact duplicate files
- duplicates crossing train/validation/test

Fix cross-split duplicates before trusting validation or test accuracy.

## 4. Train

```bash
python src/train.py
```

The model uses **MobileNetV2 transfer learning**:

1. train a new classification head while MobileNetV2 is frozen
2. fine-tune the upper MobileNetV2 layers with a smaller learning rate
3. save the validation checkpoint with the lowest validation loss

Outputs:

```text
models/spot_talent_classifier.keras
models/class_names.json
reports/training_history.json
```

## 5. Final evaluation

Only use the `test/` split for final evaluation after training decisions are finished.

```bash
python src/evaluate.py
```

It reports:

- accuracy
- macro precision
- macro recall
- macro F1-score
- per-class precision/recall/F1
- confusion matrix

Outputs:

```text
reports/evaluation.json
reports/confusion_matrix.png
```

## 6. Test one image

```bash
python src/predict.py path/to/image.jpg
```

Example behavior:

```text
Prediction: Guitar
SPOT Category: Music
Confidence: 94.20%
Needs confirmation: False

Top suggestions:
  - Guitar       94.20% (Music)
  - Piano         2.80% (Music)
  - Drums         1.50% (Music)
```

The default confidence threshold is **70%**. Below the threshold, SPOT should ask the user to confirm the category rather than silently auto-tagging the post.

## 7. API for the Android app

Run:

```bash
python src/api_server.py
```

Health check:

```text
GET /health
```

Prediction endpoint:

```text
POST /predict
```

Send `multipart/form-data` with the image under the field:

```text
image
```

Example response:

```json
{
  "success": true,
  "prediction": "Guitar",
  "raw_class": "guitar",
  "spot_category": "Music",
  "confidence": 0.942,
  "confidence_percent": 94.2,
  "recognized_talent": true,
  "needs_user_confirmation": false,
  "threshold": 0.7,
  "message": "High-confidence suggestion. The user should still be allowed to edit it.",
  "suggestions": [
    {
      "tag": "Guitar",
      "raw_class": "guitar",
      "spot_category": "Music",
      "confidence": 0.942,
      "confidence_percent": 94.2
    }
  ]
}
```

The Android Studio emulator normally reaches a Flask server running on the same PC at:

```text
http://10.0.2.2:5000
```

For a physical Android phone, both devices normally need to be on the same local network and the app must use the computer's LAN IP while testing.

## SPOT Android behavior recommendation

Do **not** silently apply the model result. Treat it as a suggestion:

```text
AI suggested:
Music · Guitar (94%)

[Use suggestion]   [Edit]
```

For low confidence or `other`:

```text
We're not sure about this post.
Choose the talent category manually.
```

This makes wrong predictions less disruptive and lets future user confirmations become useful training/feedback data.

## What to build separately later

### Personalized recommendations
Collect user interactions such as post views, watch duration, likes, comments, saves, follows, category, and timestamp. Start with a weighted/hybrid ranking system and train a recommender only after enough real interaction data exists.

### Collaboration matching
Match profile talents, requested collaborator talents, availability, preferences, and prior accepted/rejected matches. This can start as a content-based scoring algorithm before using machine learning.

### Singing
Use audio/video features rather than still-image classification.

### Feedback assistant
A safer first version is **Audience Feedback Insights**: summarize recurring strengths/suggestions from comments instead of pretending the image classifier can judge singing pitch or dance timing.

### Caption generation
Keep separate from MobileNetV2. It requires a text/generative component or a simple template-based system for V1.
