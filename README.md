# SPOT AI V1 — Visual Talent Tagging

This project is the first AI component for **SPOT**, the campus talent showcase app.
It analyzes an uploaded still image and returns **suggested visual talent tags** for the Android app.

It does **not** train SPOT's recommendation engine, collaboration matcher, feedback assistant, or caption generator. Those are separate systems with different data requirements.

## Why the V1 classes changed

V1 uses only visual classes that can reasonably be inferred from a still image:

- `singing` → Music
- `guitar` → Music
- `keyboard` → Music
- `drums` → Music
- `painting` → Art
- `drawing` → Art
- `dancing` → Dance
- `cooking` → Culinary
- `basketball` → Sports
- `futsal` → Sports
- `volleyball` → Sports
- `photography` → Media & Content Creation
- `coding` → Technology
- `other` → no supported visual talent detected

`cooking` covers both people cooking and finished or plated food.

`photography` means **a person using a camera**. It does not mean a nice-looking photo: every upload is already a photo, so the model can only learn the act of photographing. Scenic shots, portraits and selfies with no camera in view belong in `other/`.

`coding` means **code visible on a screen**, usually with someone at the keyboard. A laptop or computer with no code showing belongs in `other/`; otherwise the model learns "laptop" instead of "coding".

Sports are split into specific sports instead of one broad `sports` class, because the sports look too different from one another to learn as a single class. Add more sports the same way, each mapped to `Sports`.

`futsal` is indoor, hard-court football. Outdoor soccer on grass looks different, so keep it in `other/` unless you add a separate `soccer` class.

**Communication** (public speaking, hosting, debate) is not a class. A speaker holding a mic looks the same as a singer in a still image. Let users pick it manually, or use audio/video analysis later.

## Folder structure

```text
SPOT_AI/
├── dataset/
│   ├── train/
│   │   ├── basketball/
│   │   ├── coding/
│   │   ├── cooking/
│   │   ├── dancing/
│   │   ├── drawing/
│   │   ├── drums/
│   │   ├── futsal/
│   │   ├── guitar/
│   │   ├── keyboard/
│   │   ├── other/
│   │   ├── painting/
│   │   ├── photography/
│   │   ├── singing/
│   │   └── volleyball/
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
│   ├── moderation.py
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

Use normal images that do not clearly show one of the supported talents, for example:

- classrooms
- buildings
- animals
- ordinary selfies
- books
- cars
- landscapes
- computers with no code on screen
- unrelated campus scenes

`other` teaches the model that not every uploaded image must be forced into a talent category.

Keep food, people holding cameras, code on screens, and basketball, futsal or volleyball scenes **out** of `other/`. Those now have their own classes, and a photo that appears under two labels confuses the model. Other sports such as outdoor soccer, boxing or running can stay in `other/` until they get their own class.

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
  - Keyboard      2.80% (Music)
  - Drums         1.50% (Music)
```

The default confidence threshold is **70%**. Below the threshold, SPOT should ask the user to confirm the category rather than silently auto-tagging the post.

## 7. Explicit content filter

Every upload to `/predict` is screened for explicit, sexual or adult content **before** it is classified. SPOT uses the pre-trained [Open NSFW](https://github.com/bhky/opennsfw2) model, which returns the probability that an image is explicit. Images at or above `NSFW_BLOCK_THRESHOLD` in `src/config.py` (default **80%**) are rejected and must not be posted.

The model weights (`models/open_nsfw_weights.h5`, about 24 MB) download automatically the first time the filter runs. Include them in deployments the same way as `spot_talent_classifier.keras`, so the server does not have to download them on its first request.

Check one image:

```bash
python src/moderation.py path/to/image.jpg
```

```text
NSFW probability: 0.42%
Threshold: 80%
Blocked: False
```

On SPOT's own 5,446 dataset photos (all safe talent images), an 80% threshold flagged 14 (0.26%). Most were close-ups where hands fill the frame (typing on a laptop, playing piano) and figure drawings. Dancing, basketball and volleyball photos all scored below 80%.

| Threshold | Safe dataset photos flagged |
|---|---|
| 70% | 26 (0.48%) |
| 80% | 14 (0.26%) |
| 90% | 5 (0.09%) |

Lower the threshold to catch more borderline images at the cost of more false alarms; raise it to block only obvious cases. The filter is automated and will make some mistakes, so keep a way for users to report posts.

## 8. API for the Android app

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
  ],
  "blocked": false,
  "nsfw_probability": 0.0042,
  "nsfw_threshold": 0.8
}
```

If the image is explicit, the API returns HTTP **422** and no talent prediction:

```json
{
  "success": false,
  "error": "This image appears to contain explicit or sexual content and can't be posted.",
  "reason": "explicit_content",
  "blocked": true,
  "nsfw_probability": 0.97,
  "nsfw_threshold": 0.8
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

When the response has `"blocked": true`, do **not** upload the post. Discard the image and show the `error` message:

```text
This image can't be posted.
It appears to contain explicit or sexual content.

[Choose another photo]
```

Only upload a post after `/predict` returns `"success": true`. If the request fails for any other reason (server down, timeout), do not skip the check and upload anyway; ask the user to try again.

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
