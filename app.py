import os
import sqlite3
import torch
import gradio as gr
from groq import Groq
from torchvision import models, transforms
from datetime import datetime
import requests


# ─────────────────────────────────────────────
# 1. ENVIRONMENT CONFIG
# ─────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ─────────────────────────────────────────────
# 2. DATABASE SETUP
# ─────────────────────────────────────────────

DB_PATH = "patient_history.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS patient_sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name  TEXT,
            patient_email TEXT,
            diagnosis     TEXT NOT NULL,
            question      TEXT,
            ai_response   TEXT,
            created_at    TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()


def save_to_db(patient_name, patient_email, diagnosis, question, ai_response):
    conn = sqlite3.connect(DB_PATH)

    cur = conn.execute(
        """INSERT INTO patient_sessions
           (patient_name, patient_email, diagnosis, question, ai_response, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            patient_name,
            patient_email,
            diagnosis,
            question,
            ai_response,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    row_id = cur.lastrowid

    conn.commit()
    conn.close()

    return row_id


def get_patient_history(patient_name):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT * FROM patient_sessions WHERE patient_name = ? ORDER BY created_at DESC",
        (patient_name,)
    ).fetchall()

    conn.close()

    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# 3. EMAIL — via HTTP Webhook
# ─────────────────────────────────────────────

def send_email_summary(
    recipient_email,
    patient_name,
    diagnosis,
    question,
    ai_response
):
    webhook_url = os.getenv("EMAIL_WEBHOOK_URL", "").strip()

    if not webhook_url:
        return "⚠️ Email not configured — add EMAIL_WEBHOOK_URL in Space secrets."

    if not recipient_email or "@" not in recipient_email:
        return "⚠️ Invalid recipient email address."

    subject = f"Brain Tumor AI — Report for {patient_name or 'Patient'}"

    timestamp = datetime.now().strftime("%B %d, %Y  %H:%M")

    html_body = f"""
    <html>
    <body style="font-family:Arial,sans-serif;color:#222;max-width:600px;margin:auto">

        <div style="background:#1a1a2e;padding:20px;border-radius:8px 8px 0 0">
            <h2 style="color:#e0e0ff;margin:0">
                🧠 Brain Tumor AI — Session Summary
            </h2>

            <p style="color:#aaa;margin:4px 0 0">
                {timestamp}
            </p>
        </div>

        <div style="border:1px solid #ddd;border-top:none;padding:24px;border-radius:0 0 8px 8px">

            <table style="width:100%;border-collapse:collapse">

                <tr>
                    <td style="padding:8px 0;color:#555;width:140px">
                        <strong>Patient</strong>
                    </td>

                    <td style="padding:8px 0">
                        {patient_name or '—'}
                    </td>
                </tr>

                <tr style="background:#f9f9f9">
                    <td style="padding:8px 0;color:#555">
                        <strong>Diagnosis</strong>
                    </td>

                    <td style="padding:8px 0;color:#c0392b">
                        <strong>{diagnosis}</strong>
                    </td>
                </tr>

                <tr>
                    <td style="padding:8px 0;color:#555">
                        <strong>Question Asked</strong>
                    </td>

                    <td style="padding:8px 0">
                        {question or '—'}
                    </td>
                </tr>

            </table>

            <hr style="margin:20px 0;border:none;border-top:1px solid #eee">

            <h3 style="color:#333">
                AI Specialist Response
            </h3>

            <div style="background:#f4f4f4;padding:16px;border-radius:6px;line-height:1.6">
                {ai_response or 'No response recorded.'}
            </div>

            <p style="margin-top:24px;font-size:12px;color:#999">
                ⚠️ This is an AI-generated report. Always consult a qualified medical professional.
            </p>

        </div>

    </body>
    </html>
    """

    payload = {
        "recipient": recipient_email,
        "subject": subject,
        "html_content": html_body
    }

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=10
        )

        response.raise_for_status()

        print(f"[EMAIL] ✅ Payload sent successfully to {recipient_email}")

        return f"✅ Summary emailed to **{recipient_email}**"

    except requests.exceptions.HTTPError as e:

        print(
            f"\n--- [WEBHOOK CRASH TRACEBACK] ---\n"
            f"{e}\n"
            f"-------------------------------\n"
        )

        if "410" in str(e):
            return (
                "❌ Error: The Email Webhook URL is 'Gone'. "
                "Please generate a NEW one in Make.com."
            )

        return f"❌ Email Service Error: {e}"

    except Exception as e:

        print(
            f"\n--- [WEBHOOK CRASH TRACEBACK] ---\n"
            f"{e}\n"
            f"-------------------------------\n"
        )

        return "❌ Network error while connecting to email service."


# ─────────────────────────────────────────────
# 4. GROQ CLIENT
# ─────────────────────────────────────────────

client = None

if GROQ_API_KEY:

    try:
        client = Groq(api_key=GROQ_API_KEY)
        print("[GROQ] ✅ Client initialized")

    except Exception as e:
        print(f"[GROQ] ❌ Init error: {e}")

else:
    print("[GROQ] ❌ GROQ_API_KEY not found in environment")


# ─────────────────────────────────────────────
# 5. MODEL LOADING
# ─────────────────────────────────────────────

def load_model():

    model = models.resnet50(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        3
    )

    path = "tumor_classifier_model.pth"

    load_error = None

    if os.path.exists(path):

        if os.path.getsize(path) < 1_000_000:

            load_error = (
                f"Model file too small "
                f"({os.path.getsize(path)} bytes) — expected ~90 MB."
            )

        else:

            try:

                state_dict = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=False
                )

                model.load_state_dict(state_dict)

                print("[MODEL] ✅ Loaded successfully")

            except Exception as e:

                load_error = f"Corrupted model file: {e}"

    else:

        load_error = "tumor_classifier_model.pth not found."

    if load_error:
        print(f"[MODEL] ❌ {load_error}")

    model.eval()

    return model, load_error


brain, startup_error = load_model()


# ─────────────────────────────────────────────
# 6. CLASSIFICATION
# ─────────────────────────────────────────────

LABELS = {
    0: "No Tumor",
    1: "Benign Tumor",
    2: "Malignant Tumor"
}


def classify_tumor(image):

    if startup_error:
        return f"❌ **Model Error:** {startup_error}", None

    if image is None:
        return "⚠️ Please upload an MRI scan.", None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    try:

        img_t = transform(image).unsqueeze(0)

        with torch.no_grad():
            pred = torch.max(
                brain(img_t),
                1
            )[1].item()

        diagnosis = LABELS.get(
            pred,
            "Unknown"
        )

        return (
            f"### 🩺 Diagnosis: **{diagnosis}**",
            diagnosis
        )

    except Exception as e:

        return f"❌ Classification error: {e}", None


# ─────────────────────────────────────────────
# 7. AI CONSULTATION + SAVE + EMAIL
# ─────────────────────────────────────────────

def consult_ai(
    question,
    diagnosis,
    patient_name,
    patient_email
):

    if not diagnosis:
        return (
            "⚠️ Please classify an MRI scan first.",
            "",
            ""
        )

    if not question or not question.strip():
        return (
            "⚠️ Please type a question.",
            "",
            ""
        )

    if not client:
        return (
            "❌ Chatbot offline — GROQ_API_KEY missing in secrets.",
            "",
            ""
        )

    system_prompt = (
        f"You are an Oncology AI. "
        f"The patient's MRI result is: '{diagnosis}'. "
        "RULES: Only discuss cancer, tumors, and MRI topics. "
        "If 'No Tumor': discuss prevention and lifestyle. "
        "If 'Malignant': strongly urge immediate oncologist visit, "
        "biopsy, and treatment options. "
        "If 'Benign': advise doctor consultation, monitoring, "
        "and possible removal."
    )

    try:

        resp = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": question
                },
            ],
            model="llama-3.1-8b-instant",
        )

        ai_response = resp.choices[0].message.content

    except Exception as e:

        return (
            f"❌ Groq error: {e}",
            "",
            ""
        )

    # Save to database
    try:

        row_id = save_to_db(
            patient_name,
            patient_email,
            diagnosis,
            question,
            ai_response
        )

        db_status = (
            f"✅ Session saved to database "
            f"(Record #{row_id})"
        )

    except Exception as e:

        db_status = f"⚠️ DB save failed: {e}"

    # Send email
    if patient_email and "@" in patient_email:

        email_status = send_email_summary(
            recipient_email=patient_email,
            patient_name=patient_name,
            diagnosis=diagnosis,
            question=question,
            ai_response=ai_response,
        )

    else:

        email_status = (
            "ℹ️ No valid email provided — summary not sent."
        )

    return (
        ai_response,
        db_status,
        email_status
    )


# ─────────────────────────────────────────────
# 8. PATIENT HISTORY
# ─────────────────────────────────────────────

def load_history(patient_name):

    if not patient_name or not patient_name.strip():
        return "⚠️ Enter a patient name to load history."

    try:

        rows = get_patient_history(
            patient_name.strip()
        )

    except Exception as e:

        return f"❌ DB error: {e}"

    if not rows:

        return f"No history found for **{patient_name}**."

    lines = [
        f"## 📋 History for {patient_name} "
        f"({len(rows)} session(s))\n---"
    ]

    for r in rows:

        lines.append(
            f"**🗓 {r['created_at']}**  \n"
            f"- **Diagnosis:** {r['diagnosis']}  \n"
            f"- **Question:** {r['question'] or '—'}  \n"
            f"- **AI Response:** {r['ai_response'] or '—'}  \n"
            "---"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# 9. GRADIO UI
# ─────────────────────────────────────────────

with gr.Blocks(
    title="Brain Tumor AI Assistant"
) as demo:

    memory = gr.State(None)

    gr.Markdown(
        "# 🧠 Brain Tumor AI Assistant"
    )

    gr.Markdown(
        "Upload an MRI → get a diagnosis → consult the AI specialist → "
        "receive an email summary & have your session saved automatically."
    )

    if startup_error:

        gr.Markdown(
            f"### ❌ Model Alert: {startup_error}"
        )

    with gr.Group():

        gr.Markdown(
            "### 👤 Patient Information"
        )

        with gr.Row():

            inp_name = gr.Textbox(
                label="Patient Name",
                placeholder="e.g. John Doe"
            )

            inp_email = gr.Textbox(
                label="Patient Email",
                placeholder="e.g. john@example.com"
            )

    with gr.Row():

        with gr.Column():

            gr.Markdown(
                "### Step 1 — Analyze Scan"
            )

            img_in = gr.Image(
                type="pil",
                label="Upload MRI Scan"
            )

            btn_classify = gr.Button(
                "🔍 Classify Tumor",
                variant="primary"
            )

            out_class = gr.Markdown(
                "Results will appear here."
            )

        with gr.Column():

            gr.Markdown(
                "### Step 2 — Consult AI Specialist"
            )

            txt_in = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What should my next steps be?",
                lines=3
            )

            btn_chat = gr.Button(
                "💬 Ask AI & Save Session",
                variant="secondary"
            )

            out_chat = gr.Markdown(
                "Advice will appear here."
            )

            out_db = gr.Markdown("")

            out_email = gr.Markdown("")

    with gr.Group():

        gr.Markdown(
            "### 🗂 Patient History (from database)"
        )

        btn_history = gr.Button(
            "📂 Load History for This Patient"
        )

        out_history = gr.Markdown(
            "History will appear here."
        )

    btn_classify.click(
        classify_tumor,
        inputs=[img_in],
        outputs=[out_class, memory]
    )

    btn_chat.click(
        consult_ai,
        inputs=[
            txt_in,
            memory,
            inp_name,
            inp_email
        ],
        outputs=[
            out_chat,
            out_db,
            out_email
        ]
    )

    btn_history.click(
        load_history,
        inputs=[inp_name],
        outputs=[out_history]
    )


# ─────────────────────────────────────────────
# 10. LAUNCH
# ─────────────────────────────────────────────

demo.queue().launch(
    server_name="0.0.0.0",
    server_port=7860,
    ssr_mode=False
)
