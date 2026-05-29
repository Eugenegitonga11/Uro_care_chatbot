"""
api/index.py  —  URO-CARE RAG Chatbot
Vercel Serverless + NVIDIA API
No file system needed — knowledge base is embedded directly.
"""

import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI


EMBED_MODEL     = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL      = "meta/llama-3.1-8b-instruct"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
TOP_K           = 5
MAX_HISTORY     = 10

# ── Knowledge Base (embedded directly — no file system needed on Vercel) ─
KNOWLEDGE_BASE = [
  {
    "section": "CLINIC OVERVIEW",
    "text": "URO-CARE Urology & Andrology Center — Nairobi's Premier Urology & Andrology Center. A specialist urological and andrological center offering comprehensive diagnosis and treatment for prostate enlargement, erectile dysfunction, infertility, kidney stones, and urinary disorders for men, women and children. MOH-Licensed Medical Facility."
  },
  {
    "section": "CONTACT & LOCATION",
    "text": "Phone: +254 112 288 709. WhatsApp: https://wa.me/254112288709. Email: info@urocare.co.ke. Address: 4th Floor, PMC Building, 3rd Parklands Avenue, Nairobi, Kenya. Opening Hours: Monday to Friday 9:00 AM to 5:00 PM. Saturday 10:00 AM to 3:00 PM. Sunday Closed."
  },
  {
    "section": "ABOUT URO-CARE",
    "text": "Founded 2024 in Nairobi. 7+ years of excellence in specialist care. 5000+ patients served. 5+ specialist doctors. 98% patient satisfaction rate. 4.9 star patient rating. Mission: To revolutionise urological and andrological healthcare in Kenya by offering expert medical care, innovative treatments and compassionate patient support. Vision: To elevate urological care in Kenya to international standards. Core Values: Compassion, Confidentiality, Excellence, Accessibility. MOH-licensed facility with state-of-the-art diagnostic technology. In-house laboratory, pharmacy and imaging all under one roof. 100% confidentiality for all men's health and andrology cases. Same-week specialist appointments — no referral required. Multidisciplinary team of fellowship-trained urologists, andrologists and support specialists."
  },
  {
    "section": "SPECIALIST TEAM",
    "text": "Lead Urologist Dr JK: Specialty Urology and Andrology. Over 15 years of clinical expertise. Focus areas: Urological surgery, kidney stone management, prostate disease, men's sexual health. Fellowship trained in advanced endourology. Andrologist Dr SM: Specialty Male Reproductive Medicine. Focus areas: Male infertility, erectile dysfunction, hormonal disorders, microsurgical techniques. Fellowship in reproductive urology completed in Europe. Paediatric Urologist Dr AN: Specialty Paediatric Urology. Focus areas: Congenital anomalies, undescended testes, hypospadias, paediatric reconstructive urological surgery."
  },
  {
    "section": "SERVICES - ERECTILE DYSFUNCTION",
    "text": "Erectile Dysfunction ED and Andrology services. Comprehensive workup including hormonal panels, penile doppler ultrasound, semen analysis and varicocele assessment all done in one confidential visit. Ultrasonic Shockwave Therapy for ED: Low-intensity acoustic shockwaves restore blood flow and natural erections. Non-invasive, no surgery, no medication. 6 to 12 outpatient sessions with lasting results. Penile Implants: Inflatable and malleable penile prosthesis implant surgery for men with treatment-resistant ED. Surgery takes 45 to 90 minutes. Implants last 10 to 15 years. PRP Platelet-Rich Plasma Injection: Extracted from patient's own blood. Minimally invasive, drug-free. Requires 2 to 3 sessions. Results within 3 to 6 weeks. Pelvic Floor Training: Specialist-guided pelvic floor muscle rehabilitation. Evidence-based non-surgical first-line treatment."
  },
  {
    "section": "SERVICES - KIDNEY STONES",
    "text": "Kidney Stone treatments at URO-CARE. ESWL Ultrasonic Shockwave Stone Treatment: Non-surgical ESWL breaks kidney and ureteral stones into passable fragments. No incisions, no anaesthesia. Outpatient procedure with same-day discharge. Best for smaller kidney or upper ureteral stones. Laser and Plasma Endoscopic Treatment: Minimally invasive laser and plasma endoscopy for kidney stones and prostate enlargement. Precise same-day procedures with no external incisions and fast recovery. Laser ureteroscopy passes through natural urinary passage and is better for harder larger or lower stones. Early warning signs of kidney stones include: severe sharp pain in the back or side, pain radiating to the groin, blood in urine haematuria, nausea vomiting, frequent or burning urination."
  },
  {
    "section": "SERVICES - PROSTATE HEALTH",
    "text": "Prostate Health services at URO-CARE. Prostate and Cancer Screening: PSA blood test, digital rectal examination DRE, and prostate biopsy if indicated. Recommended from age 40 for men with family history and from age 50 for all others. African men have higher genetic risk and should not delay. Early detection saves lives. Prostate cancer is the most common cancer in Kenyan men. Prostate Enlargement BPH Treatment: Diagnosis and treatment of BPH, prostatitis and prostate cancer with evidence-based approaches including laser and plasma endoscopic treatments."
  },
  {
    "section": "SERVICES - SURGICAL AND OTHER",
    "text": "Laparoscopic Urinary Surgeries: Keyhole surgery for kidney tumours nephrectomy, ureteral obstruction pyeloplasty, bladder conditions, adrenal disorders. Benefits: smaller incisions, less pain, shorter hospital stays, faster recovery. Paediatric Urology: Specialised care for children. Conditions treated: Undescended testes, hypospadias, congenital urological conditions, paediatric reconstructive surgery. Urinary Incontinence: Comprehensive care including pelvic floor training, sling procedures and surgical repair. Treats both men and women. Artificial Urinary Sphincter AUS implantation restores voluntary bladder control. Bladder disorders, female urology, testicular urology, UTIs, varicocele, penile health conditions also treated."
  },
  {
    "section": "LABORATORY SERVICES",
    "text": "State-of-the-art in-house laboratory. Most results available same-day within 1 to 3 hours. Specialist-reviewed reports. Tests available: PSA Test results in 1 to 2 hours no fasting required avoid vigorous exercise and sexual activity 48 hours beforehand. Semen Analysis evaluates sperm count motility morphology volume pH results in 2 to 3 hours. Hormone Panel measures Testosterone FSH LH prolactin DHEA oestradiol results in 2 to 4 hours no fasting. Kidney Function Tests serum creatinine urea electrolytes eGFR results in 1 to 2 hours. Urine Analysis and Culture same-day for urinalysis 24 to 48 hours for culture. Full Blood Count results in 1 hour. Blood Glucose and HbA1c requires 8 to 12 hours fasting. STI Infection Screening for Chlamydia gonorrhoea syphilis hepatitis B and C HIV results in 2 to 4 hours. Lipid Profile requires 8 to 12 hours fasting."
  },
  {
    "section": "PHARMACY SERVICES",
    "text": "Fully stocked in-house pharmacy. Same-day dispensing immediately after consultation. Discreet plain unmarked packaging for sensitive medications. Pharmacist consultation included. All major insurers accepted with direct billing. Licensed regulated and quality-verified medications only. Medications dispensed include: Erectile Dysfunction medications PDE5 inhibitors sildenafil tadalafil vardenafil requires valid prescription. BPH and Prostate Medications alpha-blockers tamsulosin alfuzosin and 5-alpha reductase inhibitors finasteride dutasteride. Testosterone and Hormone Therapy injections gels patches clomiphene citrate. UTI and Infection Antibiotics targeted therapy based on culture results. Kidney Stone Prevention Drugs potassium citrate thiazide diuretics allopurinol. Bladder and Incontinence Medications mirabegron solifenacin. Pain Management analgesics antispasmodics anti-inflammatories. General Health Supplements zinc selenium CoQ10 antioxidants omega-3s. Post-Surgical Medications complete post-operative medication packs."
  },
  {
    "section": "INSURANCE & PAYMENTS",
    "text": "Accepted Insurance Providers at URO-CARE: Allianz Care International Insurance, Africa Medilink, Heritage Insurance Company a member of LIBERTY, Bupa, Optimum Global International Insurance Solutions, Madison Insurance. Direct billing available for most providers. Self-pay also accepted. For insurance queries call +254 112 288 709 to confirm coverage before appointment."
  },
  {
    "section": "BOOKING & APPOINTMENTS",
    "text": "How to Book an appointment at URO-CARE: Call +254 112 288 709. WhatsApp https://wa.me/254112288709. Online via website contact form. Email info@urocare.co.ke. Appointment Process: Step 1 Book Appointment call WhatsApp or book online same-week slots available no referral needed. Step 2 Consultation meet your specialist for thorough confidential consultation and examination. Step 3 Diagnosis in-house diagnostics cystoscopy ultrasound labs fast accurate results. Step 4 Treatment and Care personalised treatment plan with ongoing support and structured follow-up care. No referral required. Same-week appointments usually available. First consultation duration typically 45 to 60 minutes. Appointment confirmation team contacts patient within 24 hours of online request."
  },
  {
    "section": "FREQUENTLY ASKED QUESTIONS",
    "text": "Do I need a referral? No referral is required. You can book directly by calling +254 112 288 709 WhatsApp or online booking form. Is my consultation confidential? Yes absolutely all consultations and treatments are completely confidential. What insurance do you accept? Allianz Care Africa Medilink Heritage Insurance Bupa Optimum Global and Madison Insurance. Where is URO-CARE located? 4th Floor PMC Building 3rd Parklands Avenue Nairobi. Open Monday to Friday 9am to 5pm and Saturday 10am to 3pm. Do you treat men and women? Yes urology department treats all patients men women and children. How long does first consultation take? 45 to 60 minutes including medical history physical examination and discussion of findings. What is shockwave therapy for ED? Low-intensity shockwave therapy delivers acoustic pulses to penile tissue stimulating new blood vessel growth and improving blood flow. Requires 6 to 12 short outpatient sessions no surgery no medication. At what age should men start prostate screening? From age 40 for men with family history and from age 50 for all others. African men have higher genetic risk. Is the pharmacy confidential? Yes 100 percent sensitive prescriptions dispensed in plain unmarked packaging."
  },
  {
    "section": "KEY DIFFERENTIATORS",
    "text": "Why choose URO-CARE: Fellowship-Trained Specialists with international fellowships and global best practices. Absolute Confidentiality privacy guaranteed for all consultations especially sensitive andrology cases. Advanced In-House Diagnostics urodynamics flexible cystoscopy ultrasound and full laboratory with fast accurate results. One-Stop Centre consultation diagnostics pharmacy and treatment all under one roof at 3rd Parklands Avenue Nairobi. Same-Week Appointments no referral required minimal waiting. No Referral Needed patients book directly. Insurance Accepted 6 plus major providers including international insurers. Male infertility treatment available including semen analysis hormonal profiling varicocele repair and surgical sperm retrieval. Shockwave therapy for ED available in Nairobi. URO-CARE is one of the few clinics in Kenya offering low-intensity ultrasonic shockwave therapy LiSWT for erectile dysfunction."
  }
]

SYSTEM_PROMPT = """You are a warm, professional and knowledgeable patient care assistant for URO-CARE Urology & Andrology Center — Nairobi's premier specialist clinic.

You answer questions ONLY using the context provided below from the URO-CARE knowledge base. If the answer is not in the context, say you don't have that specific information and direct the patient to call +254 112 288 709 or WhatsApp for personalised help.

Core rules:
1. NEVER provide a medical diagnosis or prescribe treatment.
2. Keep replies concise, warm and human — 2-5 sentences for simple questions, slightly longer for complex ones.
3. Always end with a helpful next step (book an appointment, call us, WhatsApp us).
4. Use plain language — no jargon unless explaining a term the patient asked about.
5. Be reassuring and empathetic, especially for sensitive topics (ED, infertility, STIs).
6. For emergencies or urgent symptoms, always advise calling +254 112 288 709 immediately.

Contact info to always have ready:
- Phone / WhatsApp: +254 112 288 709
- Email: info@urocare.co.ke
- Location: 4th Floor, PMC Building, 3rd Parklands Avenue, Nairobi
- Hours: Mon-Fri 9 AM-5 PM | Sat 10 AM-3 PM
- No referral needed. Same-week appointments available. 100% confidential.

--- KNOWLEDGE BASE CONTEXT ---
{context}
--- END CONTEXT ---"""

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), '../..', 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), '../..', 'static')
)
CORS(app)

_client = None
_embeddings = None  # cached embeddings for knowledge base


def get_client():
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=key, base_url=NVIDIA_BASE_URL)
    return _client


def get_kb_embeddings():
    """Embed the knowledge base once and cache it."""
    global _embeddings
    if _embeddings is None:
        texts = [item["text"] for item in KNOWLEDGE_BASE]
        resp  = get_client().embeddings.create(
            model=EMBED_MODEL,
            input=texts,
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "END"}
        )
        _embeddings = [item.embedding for item in resp.data]
    return _embeddings


def cosine_sim(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def retrieve(query, k=TOP_K):
    resp = get_client().embeddings.create(
        model=EMBED_MODEL,
        input=[query],
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "END"}
    )
    q_emb   = resp.data[0].embedding
    kb_embs = get_kb_embeddings()

    scored = []
    for i, item in enumerate(KNOWLEDGE_BASE):
        score = cosine_sim(q_emb, kb_embs[i])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]

    context = "\n\n---\n\n".join(
        f"[Section: {item['section']} | Relevance: {round(s*100,1)}%]\n{item['text']}"
        for s, item in top
    )
    sources = []
    seen    = set()
    for s, item in top:
        if s > 0.3 and item["section"] not in seen:
            seen.add(item["section"])
            sources.append({"section": item["section"], "relevance": round(s*100,1)})

    return context, sources


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    try:
        return jsonify({
            "status":       "ok",
            "chunks_in_db": len(KNOWLEDGE_BASE),
            "chat_model":   CHAT_MODEL,
            "embed_model":  EMBED_MODEL,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 200


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data    = request.get_json(silent=True) or {}
        message = (data.get("message") or "").strip()
        history = data.get("history") or []

        if not message:
            return jsonify({"error": "No message"}), 400

        context, sources = retrieve(message)
        system   = SYSTEM_PROMPT.format(context=context)
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-(MAX_HISTORY * 2):])
        messages.append({"role": "user", "content": message})

        resp  = get_client().chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=600,
        )
        reply = resp.choices[0].message.content
        return jsonify({"reply": reply, "sources": sources})

    except Exception as e:
        return jsonify({
            "reply":   "Technical issue. Please call +254 112 288 709!",
            "sources": []
        }), 200


# Vercel needs this
app_handler = app
