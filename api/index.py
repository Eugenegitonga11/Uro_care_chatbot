"""
api/index.py  —  URO-CARE RAG Chatbot — Vercel Serverless
"""

import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL     = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL      = "meta/llama-3.1-8b-instruct"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
TOP_K           = 5
MAX_HISTORY     = 10

# ── Knowledge Base ─────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
  {"section":"CLINIC OVERVIEW","text":"URO-CARE Urology & Andrology Center — Nairobi's Premier Urology & Andrology Center. A specialist urological and andrological center offering comprehensive diagnosis and treatment for prostate enlargement, erectile dysfunction, infertility, kidney stones, and urinary disorders for men, women and children. MOH-Licensed Medical Facility."},
  {"section":"CONTACT & LOCATION","text":"Phone: +254 112 288 709. WhatsApp: https://wa.me/254112288709. Email: info@urocare.co.ke. Address: 4th Floor, PMC Building, 3rd Parklands Avenue, Nairobi, Kenya. Opening Hours: Monday to Friday 9:00 AM to 5:00 PM. Saturday 10:00 AM to 3:00 PM. Sunday Closed."},
  {"section":"ABOUT URO-CARE","text":"Founded 2024 in Nairobi. 7+ years of excellence in specialist care. 5000+ patients served. 5+ specialist doctors. 98% patient satisfaction rate. 4.9 star patient rating. Mission: To revolutionise urological and andrological healthcare in Kenya. MOH-licensed facility with state-of-the-art diagnostic technology. In-house laboratory, pharmacy and imaging all under one roof. 100% confidentiality for all men's health and andrology cases. Same-week specialist appointments no referral required. Multidisciplinary team of fellowship-trained urologists andrologists and support specialists."},
  {"section":"SPECIALIST TEAM","text":"Lead Urologist Dr JK: Specialty Urology and Andrology. Over 15 years clinical expertise. Focus: Urological surgery, kidney stone management, prostate disease, men's sexual health. Fellowship trained in advanced endourology. Andrologist Dr SM: Specialty Male Reproductive Medicine. Focus: Male infertility, erectile dysfunction, hormonal disorders, microsurgical techniques. Fellowship in reproductive urology completed in Europe. Paediatric Urologist Dr AN: Specialty Paediatric Urology. Focus: Congenital anomalies, undescended testes, hypospadias, paediatric reconstructive urological surgery."},
  {"section":"SERVICES - ERECTILE DYSFUNCTION","text":"ED and Andrology services. Comprehensive workup including hormonal panels, penile doppler ultrasound, semen analysis and varicocele assessment all in one confidential visit. Ultrasonic Shockwave Therapy for ED: Low-intensity acoustic shockwaves restore blood flow and natural erections. Non-invasive, no surgery, no medication. 6 to 12 outpatient sessions. Penile Implants: Inflatable and malleable penile prosthesis for treatment-resistant ED. Surgery 45 to 90 minutes. Implants last 10 to 15 years. PRP Injection: From patient's own blood. Minimally invasive drug-free. 2 to 3 sessions. Results within 3 to 6 weeks. Pelvic Floor Training: Specialist-guided rehabilitation. Evidence-based non-surgical first-line treatment."},
  {"section":"SERVICES - KIDNEY STONES","text":"Kidney Stone treatments. ESWL Shockwave: Non-surgical breaks stones into passable fragments. No incisions no anaesthesia. Outpatient same-day discharge. Best for smaller upper ureteral stones. Laser Endoscopic Treatment: Minimally invasive laser and plasma endoscopy. Precise same-day procedures no external incisions fast recovery. Better for harder larger lower stones. Early warning signs: severe sharp back or side pain, pain radiating to groin, blood in urine, nausea vomiting, frequent or burning urination."},
  {"section":"SERVICES - PROSTATE HEALTH","text":"Prostate Health services. Prostate and Cancer Screening: PSA blood test, digital rectal examination DRE, prostate biopsy if indicated. Recommended from age 40 for men with family history and from age 50 for all others. African men have higher genetic risk and should not delay. Early detection saves lives. Prostate cancer is the most common cancer in Kenyan men. Prostate Enlargement BPH Treatment: Diagnosis and treatment of BPH prostatitis and prostate cancer with evidence-based approaches including laser and plasma endoscopic treatments."},
  {"section":"SERVICES - SURGICAL AND OTHER","text":"Laparoscopic Urinary Surgeries: Keyhole surgery for kidney tumours, ureteral obstruction, bladder conditions, adrenal disorders. Benefits: smaller incisions less pain shorter hospital stays faster recovery. Paediatric Urology: Undescended testes hypospadias congenital urological conditions paediatric reconstructive surgery. Urinary Incontinence: Pelvic floor training sling procedures surgical repair. Treats men and women. Artificial Urinary Sphincter AUS implantation restores voluntary bladder control. Bladder disorders female urology testicular urology UTIs varicocele penile health conditions also treated."},
  {"section":"LABORATORY SERVICES","text":"State-of-the-art in-house laboratory. Most results same-day within 1 to 3 hours. Tests: PSA Test 1 to 2 hours no fasting avoid vigorous exercise and sexual activity 48 hours beforehand. Semen Analysis sperm count motility morphology volume 2 to 3 hours. Hormone Panel Testosterone FSH LH prolactin DHEA oestradiol 2 to 4 hours no fasting. Kidney Function Tests serum creatinine urea electrolytes eGFR 1 to 2 hours. Urine Analysis and Culture same-day for urinalysis 24 to 48 hours for culture. Full Blood Count 1 hour. Blood Glucose and HbA1c requires 8 to 12 hours fasting. STI Screening Chlamydia gonorrhoea syphilis hepatitis B C HIV 2 to 4 hours. Lipid Profile requires 8 to 12 hours fasting."},
  {"section":"PHARMACY SERVICES","text":"Fully stocked in-house pharmacy. Same-day dispensing after consultation. Discreet plain unmarked packaging for sensitive medications. Pharmacist consultation included. All major insurers accepted direct billing available. Medications: ED medications PDE5 inhibitors sildenafil tadalafil vardenafil requires prescription. BPH Prostate Medications tamsulosin alfuzosin finasteride dutasteride. Testosterone Hormone Therapy injections gels patches. UTI Antibiotics targeted based on culture results. Kidney Stone Prevention potassium citrate thiazide diuretics allopurinol. Bladder Incontinence Medications mirabegron solifenacin. Pain Management analgesics antispasmodics. Supplements zinc selenium CoQ10 antioxidants omega-3s. Post-Surgical Medication packs."},
  {"section":"INSURANCE & PAYMENTS","text":"Accepted Insurance at URO-CARE: Allianz Care International Insurance, Africa Medilink, Heritage Insurance Company a member of LIBERTY, Bupa, Optimum Global International Insurance Solutions, Madison Insurance. Direct billing available. Self-pay accepted. For insurance queries call +254 112 288 709."},
  {"section":"BOOKING & APPOINTMENTS","text":"How to Book: Call +254 112 288 709. WhatsApp https://wa.me/254112288709. Online via website. Email info@urocare.co.ke. Appointment Process: Book - call WhatsApp or online same-week slots no referral needed. Consultation - meet specialist thorough confidential examination. Diagnosis - in-house diagnostics cystoscopy ultrasound labs fast accurate results. Treatment - personalised treatment plan ongoing support structured follow-up. No referral required. Same-week appointments available. First consultation 45 to 60 minutes. Confirmation within 24 hours of online request."},
  {"section":"FREQUENTLY ASKED QUESTIONS","text":"Do I need a referral? No referral required book directly by calling +254 112 288 709 WhatsApp or online. Is consultation confidential? Yes absolutely all consultations and treatments completely confidential. What insurance accepted? Allianz Care Africa Medilink Heritage Insurance Bupa Optimum Global Madison Insurance. Where is URO-CARE? 4th Floor PMC Building 3rd Parklands Avenue Nairobi. Open Monday to Friday 9am to 5pm Saturday 10am to 3pm. Do you treat men and women? Yes urology treats all patients men women and children. How long does first consultation take? 45 to 60 minutes. What is shockwave therapy for ED? Low-intensity shockwave therapy delivers acoustic pulses stimulating new blood vessel growth improving blood flow. 6 to 12 outpatient sessions no surgery no medication. At what age prostate screening? Age 40 with family history age 50 for all others. Is pharmacy confidential? Yes 100 percent sensitive prescriptions in plain unmarked packaging."},
  {"section":"KEY DIFFERENTIATORS","text":"Why choose URO-CARE: Fellowship-Trained Specialists international fellowships global best practices. Absolute Confidentiality privacy guaranteed. Advanced In-House Diagnostics urodynamics flexible cystoscopy ultrasound full laboratory. One-Stop Centre consultation diagnostics pharmacy treatment all under one roof 3rd Parklands Avenue Nairobi. Same-Week Appointments no referral required minimal waiting. Insurance Accepted 6 plus major providers including international insurers. Male infertility treatment semen analysis hormonal profiling varicocele repair surgical sperm retrieval. Shockwave therapy LiSWT for ED available — one of few clinics in Kenya offering this."}
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

# ── HTML template as a string (no file system needed) ─────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>URO-CARE | AI Patient Assistant</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--cream:#F8F4EF;--navy:#0A1F44;--blue:#1B5FA8;--sky:#3A8FCA;--text:#1A1A2A;--muted:#6B7280;--border:#E5DDD4;--white:#FFFFFF}
html{scroll-behavior:smooth}
body{font-family:'Outfit',sans-serif;color:var(--text);background:var(--cream);min-height:100vh}
.page-bg{min-height:100vh;display:flex;align-items:center;justify-content:center;flex-direction:column;background:radial-gradient(ellipse at 20% 50%,rgba(26,95,168,0.08) 0%,transparent 60%),radial-gradient(ellipse at 80% 20%,rgba(196,154,60,0.06) 0%,transparent 50%),var(--cream);padding:60px 24px 160px;text-align:center}
.page-hero-eyebrow{font-size:11px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:var(--blue);margin-bottom:16px}
.page-hero-title{font-family:'Cormorant Garamond',serif;font-size:clamp(2.4rem,5vw,4rem);font-weight:600;color:var(--navy);line-height:1.12;margin-bottom:18px}
.page-hero-title em{color:var(--sky);font-style:italic}
.page-hero-sub{color:var(--muted);font-size:1rem;line-height:1.75;max-width:480px;margin:0 auto 36px;font-weight:300}
.page-badges{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-bottom:12px}
.badge{display:inline-flex;align-items:center;gap:6px;background:var(--white);border:1px solid var(--border);border-radius:20px;padding:6px 14px;font-size:12.5px;font-weight:500;color:var(--navy)}
.badge svg{width:13px;height:13px;stroke:var(--sky);fill:none;stroke-width:2}
#status-bar{position:fixed;top:0;left:0;right:0;background:var(--navy);color:rgba(255,255,255,0.75);font-size:12px;padding:8px 20px;display:flex;align-items:center;justify-content:space-between;z-index:800;gap:12px}
#status-bar .sb-left{display:flex;align-items:center;gap:8px}
#status-bar .sb-dot{width:7px;height:7px;border-radius:50%;background:#6b7280;flex-shrink:0}
#status-bar .sb-dot.ok{background:#4ade80}
#status-bar .sb-dot.err{background:#f87171}
.float-wa{position:fixed;bottom:100px;right:28px;width:54px;height:54px;background:#25D366;border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 20px rgba(37,211,102,0.45);z-index:900;text-decoration:none;transition:all 0.3s}
.float-wa:hover{transform:scale(1.12)}
.float-wa svg{width:28px;height:28px}
.float-wa-tooltip{position:absolute;right:64px;background:#1a1a2a;color:white;font-size:12px;font-weight:500;padding:6px 12px;border-radius:6px;white-space:nowrap;opacity:0;pointer-events:none;transition:opacity 0.25s}
.float-wa-tooltip::after{content:'';position:absolute;left:100%;top:50%;transform:translateY(-50%);border:5px solid transparent;border-left-color:#1a1a2a}
.float-wa:hover .float-wa-tooltip{opacity:1}
.chat-fab{position:fixed;bottom:28px;right:28px;width:58px;height:58px;background:linear-gradient(135deg,#0A1F44 0%,#1B5FA8 60%,#3A8FCA 100%);border-radius:50%;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 20px rgba(26,95,168,0.5);z-index:901;cursor:pointer;transition:all 0.3s;user-select:none}
.chat-fab:hover{transform:scale(1.1)}
.chat-fab-icon{display:flex;align-items:center;justify-content:center}
.chat-fab-icon svg{width:26px;height:26px}
.chat-fab-badge{position:absolute;top:-3px;right:-3px;width:20px;height:20px;background:#e53935;color:white;font-size:11px;font-weight:700;border-radius:50%;display:flex;align-items:center;justify-content:center;border:2px solid white;opacity:0;transform:scale(0.4);animation:badgePop 0.5s cubic-bezier(0.34,1.56,0.64,1) 3s forwards}
@keyframes badgePop{to{opacity:1;transform:scale(1)}}
.chat-window{position:fixed;bottom:100px;right:28px;width:380px;max-height:580px;background:white;border-radius:20px;box-shadow:0 32px 80px rgba(10,31,68,0.22),0 4px 16px rgba(10,31,68,0.08);display:flex;flex-direction:column;z-index:902;overflow:hidden;transform:scale(0.88) translateY(24px);transform-origin:bottom right;opacity:0;pointer-events:none;transition:all 0.38s cubic-bezier(0.34,1.56,0.64,1);font-family:'Outfit',sans-serif}
.chat-window.open{transform:scale(1) translateY(0);opacity:1;pointer-events:all}
.chat-header{background:linear-gradient(135deg,#0A1F44 0%,#1B5FA8 100%);padding:16px 18px;display:flex;align-items:center;gap:12px;flex-shrink:0}
.chat-header-avatar{width:44px;height:44px;border-radius:50%;background:rgba(255,255,255,0.15);display:flex;align-items:center;justify-content:center;flex-shrink:0}
.chat-header-info{flex:1;min-width:0}
.chat-header-name{color:white;font-weight:700;font-size:14.5px}
.chat-header-status{display:flex;align-items:center;gap:6px;color:rgba(255,255,255,0.72);font-size:11.5px;margin-top:2px}
.status-dot{width:7px;height:7px;background:#4ade80;border-radius:50%;flex-shrink:0;animation:sdPulse 2s ease-in-out infinite}
@keyframes sdPulse{0%,100%{opacity:1}50%{opacity:0.4}}
.chat-header-close{background:rgba(255,255,255,0.12);border:none;border-radius:50%;width:32px;height:32px;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:background 0.2s;flex-shrink:0}
.chat-header-close:hover{background:rgba(255,255,255,0.26)}
.chat-header-close svg{width:14px;height:14px}
.chat-powered{background:rgba(255,255,255,0.08);border-top:1px solid rgba(255,255,255,0.08);padding:5px 18px;font-size:10px;color:rgba(255,255,255,0.45);letter-spacing:0.3px;text-align:right;flex-shrink:0}
.chat-messages{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px;background:#F8F4EF;scroll-behavior:smooth}
.chat-messages::-webkit-scrollbar{width:4px}
.chat-messages::-webkit-scrollbar-thumb{background:#ddd;border-radius:4px}
.chat-msg{display:flex;gap:8px;align-items:flex-end;animation:msgIn 0.28s ease}
@keyframes msgIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.chat-msg.bot{flex-direction:row}
.chat-msg.user{flex-direction:row-reverse}
.chat-msg-avatar{width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,#0A1F44,#1B5FA8);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:9px;font-weight:800;color:white;letter-spacing:0.5px}
.chat-bubble{max-width:78%;padding:11px 14px;border-radius:16px;font-size:13.5px;line-height:1.6;word-break:break-word}
.chat-msg.bot .chat-bubble{background:white;color:#1A1A2A;border-bottom-left-radius:4px;box-shadow:0 1px 6px rgba(0,0,0,0.08)}
.chat-msg.user .chat-bubble{background:linear-gradient(135deg,#1B5FA8,#3A8FCA);color:white;border-bottom-right-radius:4px}
.chat-time{font-size:10px;color:#9CA3AF;padding:2px 4px}
.chat-msg.user .chat-time{text-align:right}
.chat-sources{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.src-pill{font-size:10px;color:var(--muted);background:rgba(26,95,168,0.07);border:1px solid rgba(26,95,168,0.15);border-radius:10px;padding:2px 8px}
.typing-bubble{background:white;border-radius:16px;border-bottom-left-radius:4px;padding:12px 15px;box-shadow:0 1px 6px rgba(0,0,0,0.08);display:flex;gap:5px;align-items:center}
.t-dot{width:7px;height:7px;background:#9CA3AF;border-radius:50%;animation:tBounce 1.2s ease-in-out infinite}
.t-dot:nth-child(2){animation-delay:0.2s}
.t-dot:nth-child(3){animation-delay:0.4s}
@keyframes tBounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-7px)}}
.chat-quick-replies{padding:10px 14px;display:flex;gap:7px;flex-wrap:wrap;background:white;border-top:1px solid #F0EEE9;flex-shrink:0}
.qr-btn{background:#EEF4FC;color:#1B5FA8;border:1.5px solid #C8DDF5;border-radius:20px;padding:5px 13px;font-size:12px;font-weight:600;cursor:pointer;transition:all 0.2s;font-family:'Outfit',sans-serif;white-space:nowrap}
.qr-btn:hover{background:#1B5FA8;color:white;border-color:#1B5FA8}
.chat-input-row{display:flex;gap:8px;padding:12px 14px;background:white;border-top:1px solid #E5DDD4;flex-shrink:0;align-items:center}
.chat-input{flex:1;border:1.5px solid #E5DDD4;border-radius:22px;padding:9px 15px;font-size:13.5px;font-family:'Outfit',sans-serif;outline:none;transition:border-color 0.2s;color:#1A1A2A;background:#FDFBF8}
.chat-input:focus{border-color:#3A8FCA}
.chat-input:disabled{opacity:0.6;cursor:not-allowed}
.chat-send-btn{width:40px;height:40px;flex-shrink:0;background:linear-gradient(135deg,#1B5FA8,#3A8FCA);border:none;border-radius:50%;display:flex;align-items:center;justify-content:center;cursor:pointer;transition:all 0.25s}
.chat-send-btn:hover:not(:disabled){transform:scale(1.08)}
.chat-send-btn:disabled{opacity:0.5;cursor:not-allowed}
.chat-send-btn svg{width:16px;height:16px}
@media(max-width:480px){.chat-window{width:calc(100vw - 16px);right:8px;bottom:96px;max-height:75vh}.float-wa{right:12px;bottom:92px}.chat-fab{right:12px;bottom:24px}}
</style>
</head>
<body>
<div id="status-bar">
  <div class="sb-left"><div class="sb-dot" id="sbDot"></div><span id="sbText">Connecting to URO-CARE AI…</span></div>
  <span style="opacity:0.4">RAG · NVIDIA · Llama 3.1</span>
</div>
<div class="page-bg" style="padding-top:80px;">
  <p class="page-hero-eyebrow">AI-Powered Patient Assistant</p>
  <h1 class="page-hero-title">Ask URO-CARE<br><em>Anything</em></h1>
  <p class="page-hero-sub">Our intelligent assistant answers your questions about urology, andrology, kidney stones, prostate health, appointments and more — powered by our verified knowledge base.</p>
  <div class="page-badges">
    <span class="badge"><svg viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>100% Confidential</span>
    <span class="badge"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>Available 24/7</span>
    <span class="badge"><svg viewBox="0 0 24 24"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>4th Floor, PMC, Parklands</span>
    <span class="badge"><svg viewBox="0 0 24 24"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07A19.5 19.5 0 0 1 4.69 12 19.79 19.79 0 0 1 1.61 3.38 2 2 0 0 1 3.59 1.18h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L7.91 8.73a16 16 0 0 0 6.29 6.29l.92-.92a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/></svg>+254 112 288 709</span>
  </div>
  <p style="color:var(--muted);font-size:12px;margin-top:8px;">👇 Click the blue chat button in the bottom-right corner to get started</p>
</div>
<a href="https://wa.me/254112288709" class="float-wa" target="_blank" rel="noopener" aria-label="WhatsApp">
  <svg viewBox="0 0 24 24"><path fill="white" d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z"/></svg>
  <span class="float-wa-tooltip">Chat on WhatsApp</span>
</a>
<div id="chat-fab" class="chat-fab" onclick="toggleChat()" aria-label="Open chat" role="button" tabindex="0">
  <div class="chat-fab-icon" id="fabOpen">
    <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
  </div>
  <div class="chat-fab-icon" id="fabClose" style="display:none">
    <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
  </div>
  <div class="chat-fab-badge" id="chatBadge">1</div>
</div>
<div id="chat-window" class="chat-window" role="dialog" aria-label="URO-CARE Chat">
  <div class="chat-header">
    <div class="chat-header-avatar">
      <svg viewBox="0 0 40 40" fill="none" style="width:30px;height:30px"><circle cx="20" cy="20" r="20" fill="rgba(255,255,255,0.15)"/><path d="M20 8C13.4 8 8 13.4 8 20s5.4 12 12 12 12-5.4 12-12S26.6 8 20 8zm0 5c2.2 0 4 1.8 4 4s-1.8 4-4 4-4-1.8-4-4 1.8-4 4-4zm0 17c-3 0-5.7-1.5-7.4-3.8.1-2.4 4.9-3.7 7.4-3.7 2.5 0 7.3 1.3 7.4 3.7C25.7 28.5 23 30 20 30z" fill="white"/></svg>
    </div>
    <div class="chat-header-info">
      <div class="chat-header-name">URO-CARE Assistant</div>
      <div class="chat-header-status"><span class="status-dot"></span>Online — powered by AI</div>
    </div>
    <button class="chat-header-close" onclick="toggleChat()" aria-label="Close">
      <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
  </div>
  <div class="chat-powered">Powered by RAG · NVIDIA · Llama 3.1</div>
  <div class="chat-messages" id="chatMessages"></div>
  <div class="chat-quick-replies" id="quickReplies">
    <button class="qr-btn" onclick="sendQuickReply('Tell me about kidney stone treatment options')">🫘 Kidney Stones</button>
    <button class="qr-btn" onclick="sendQuickReply('What prostate health services do you offer?')">🩺 Prostate Health</button>
    <button class="qr-btn" onclick="sendQuickReply('How do I book an appointment?')">📅 Book Appointment</button>
    <button class="qr-btn" onclick="sendQuickReply('What are your opening hours and location?')">📍 Location & Hours</button>
    <button class="qr-btn" onclick="sendQuickReply('Do you accept insurance? Which providers?')">🛡️ Insurance</button>
    <button class="qr-btn" onclick="sendQuickReply('What laboratory tests do you offer?')">🔬 Lab Tests</button>
  </div>
  <div class="chat-input-row">
    <input type="text" id="chatInput" class="chat-input" placeholder="Ask anything about URO-CARE…" onkeydown="if(event.key==='Enter'&&!event.shiftKey)sendMessage()" maxlength="500" autocomplete="off"/>
    <button class="chat-send-btn" id="sendBtn" onclick="sendMessage()" aria-label="Send">
      <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
    </button>
  </div>
</div>
<script>
(function(){
  var chatOpen=false,chatReady=false,isBusy=false,history=[];
  function el(id){return document.getElementById(id)}
  function now(){return new Date().toLocaleTimeString('en-KE',{hour:'2-digit',minute:'2-digit'})}
  function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
  function scrollBottom(){var m=el('chatMessages');if(m)m.scrollTop=m.scrollHeight}
  function setBusy(v){isBusy=v;var b=el('sendBtn'),i=el('chatInput');if(b)b.disabled=v;if(i)i.disabled=v}
  function hideQR(){var q=el('quickReplies');if(!q||q.style.display==='none')return;q.style.transition='opacity 0.2s';q.style.opacity='0';setTimeout(function(){q.style.display='none'},200)}
  function fmt(t){t=t.replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>');t=t.replace(/(\\+254[\\s\\d]{9,13})/g,'<a href="tel:$1" style="color:#1B5FA8;font-weight:600">$1</a>');t=t.replace(/\\n\\n/g,'<br><br>').replace(/\\n/g,'<br>');return t}
  function checkHealth(){
    fetch('/health').then(function(r){return r.json()}).then(function(d){
      if(d.status==='ok'){el('sbDot').className='sb-dot ok';el('sbText').textContent='URO-CARE AI ready · '+d.chunks_in_db+' knowledge chunks loaded'}
      else throw new Error(d.message)
    }).catch(function(){el('sbDot').className='sb-dot err';el('sbText').textContent='AI unavailable — call +254 112 288 709'})
  }
  function addTyping(){var d=el('chatMessages'),e=document.createElement('div');e.className='chat-msg bot';e.id='typing-ind';e.innerHTML='<div class="chat-msg-avatar">UC</div><div class="typing-bubble"><div class="t-dot"></div><div class="t-dot"></div><div class="t-dot"></div></div>';d.appendChild(e);scrollBottom()}
  function removeTyping(){var t=el('typing-ind');if(t)t.remove()}
  function addUserMsg(text){var d=el('chatMessages'),e=document.createElement('div');e.className='chat-msg user';e.innerHTML='<div><div class="chat-bubble">'+esc(text)+'</div><div class="chat-time">'+now()+'</div></div>';d.appendChild(e);scrollBottom()}
  function addBotMsg(html,sources){
    sources=Array.isArray(sources)?sources:[];
    var srcHtml=sources.length?'<div class="chat-sources">'+sources.map(function(s){return'<span class="src-pill">&#128218; '+s.section+'</span>'}).join('')+'</div>':'';
    var d=el('chatMessages'),e=document.createElement('div');e.className='chat-msg bot';
    e.innerHTML='<div class="chat-msg-avatar">UC</div><div><div class="chat-bubble">'+html+'</div><div class="chat-time">'+now()+'</div>'+srcHtml+'</div>';
    d.appendChild(e);scrollBottom()
  }
  function fetchChat(userText){
    setBusy(true);addTyping();
    fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:userText,history:history.slice(-20)})})
    .then(function(r){return r.json()})
    .then(function(d){
      removeTyping();
      var reply=d.reply||'Sorry I could not get a response. Please call +254 112 288 709.';
      addBotMsg(fmt(reply),d.sources||[]);
      history.push({role:'user',content:userText});
      history.push({role:'assistant',content:reply})
    })
    .catch(function(){removeTyping();addBotMsg('Technical issue. Please call <strong>+254 112 288 709</strong>.',[])})
    .finally(function(){setBusy(false);var i=el('chatInput');if(i)i.focus()})
  }
  function startChat(){el('chatMessages').innerHTML='';history=[];addTyping();setTimeout(function(){removeTyping();addBotMsg('Welcome to <strong>URO-CARE</strong>! &#128075; I am your AI patient care assistant, powered by our verified knowledge base.<br><br>I can answer questions about services, lab tests, pharmacy, insurance, appointments and more. How can I help you today?',[]);history.push({role:'assistant',content:'Welcome to URO-CARE! How can I help you today?'})},1200)}
  window.toggleChat=function(){
    chatOpen=!chatOpen;var w=el('chat-window'),fo=el('fabOpen'),fc=el('fabClose'),b=el('chatBadge');
    if(chatOpen){w.classList.add('open');if(fo)fo.style.display='none';if(fc)fc.style.display='flex';if(b)b.style.display='none';if(!chatReady){chatReady=true;startChat()}setTimeout(function(){var i=el('chatInput');if(i)i.focus()},400)}
    else{w.classList.remove('open');if(fo)fo.style.display='flex';if(fc)fc.style.display='none'}
  };
  window.sendMessage=function(){if(isBusy)return;var i=el('chatInput'),text=i?i.value.trim():'';if(!text)return;i.value='';hideQR();addUserMsg(text);fetchChat(text)};
  window.sendQuickReply=function(text){if(isBusy)return;hideQR();addUserMsg(text);fetchChat(text)};
  setTimeout(function(){if(!chatOpen){var b=el('chatBadge');if(b){b.style.opacity='1';b.style.transform='scale(1)'}}},3000);
  checkHealth();
})();
</script>
</body>
</html>"""

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

_client     = None
_kb_embeds  = None


def get_client():
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=key, base_url=NVIDIA_BASE_URL)
    return _client


def get_kb_embeds():
    global _kb_embeds
    if _kb_embeds is None:
        texts = [item["text"] for item in KNOWLEDGE_BASE]
        resp  = get_client().embeddings.create(
            model=EMBED_MODEL, input=texts,
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "END"}
        )
        _kb_embeds = [item.embedding for item in resp.data]
    return _kb_embeds


def cosine(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def retrieve(query):
    q_emb = get_client().embeddings.create(
        model=EMBED_MODEL, input=[query],
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "END"}
    ).data[0].embedding

    scored = sorted(
        [(cosine(q_emb, e), item) for e, item in zip(get_kb_embeds(), KNOWLEDGE_BASE)],
        key=lambda x: x[0], reverse=True
    )[:TOP_K]

    context = "\n\n---\n\n".join(
        f"[Section: {item['section']} | Relevance: {round(s*100,1)}%]\n{item['text']}"
        for s, item in scored
    )
    seen, sources = set(), []
    for s, item in scored:
        if s > 0.3 and item["section"] not in seen:
            seen.add(item["section"])
            sources.append({"section": item["section"], "relevance": round(s*100,1)})
    return context, sources


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return HTML_PAGE, 200, {"Content-Type": "text/html"}


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "chunks_in_db": len(KNOWLEDGE_BASE),
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
    })


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
            model=CHAT_MODEL, messages=messages, temperature=0.4, max_tokens=600
        )
        return jsonify({"reply": resp.choices[0].message.content, "sources": sources})

    except Exception as e:
        return jsonify({"reply": "Technical issue. Please call +254 112 288 709!", "sources": []}), 200


# Vercel handler
app_handler = app