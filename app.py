
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import time
import json
import os

st.set_page_config(page_title="MediRAG AI", page_icon="🧬", layout="wide")

if "dark_mode" not in st.session_state: st.session_state.dark_mode = False
if "messages" not in st.session_state: st.session_state.messages = []
if "pending" not in st.session_state: st.session_state.pending = ""
if "ikey" not in st.session_state: st.session_state.ikey = 0
if "lang" not in st.session_state: st.session_state.lang = "English"
if "feedback" not in st.session_state: st.session_state.feedback = {}

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
HISTORY_PATH = "data/chat_history.json"
LANGUAGES = {
    "English":"en-US","Hindi":"hi-IN","Telugu":"te-IN","Tamil":"ta-IN",
    "Kannada":"kn-IN","Malayalam":"ml-IN","Spanish":"es-ES","French":"fr-FR",
    "Arabic":"ar-SA","German":"de-DE",
}

def save_history():
    try:
        with open(HISTORY_PATH, "w") as f:
            json.dump(st.session_state.messages, f)
    except: pass

def load_history():
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r") as f:
                return json.load(f)
    except: pass
    return []

if not st.session_state.messages:
    loaded = load_history()
    if loaded:
        st.session_state.messages = loaded

@st.cache_resource
def load_all():
    c = Groq(api_key=GROQ_API_KEY)
    e = SentenceTransformer("all-MiniLM-L6-v2")
    idx = faiss.read_index("data/faiss_index.bin")
    with open("data/documents.pkl","rb") as f:
        docs = pickle.load(f)
    return c, e, idx, docs

client, embed_model, index, documents = load_all()

def ask(question, lang_name):
    qv = np.array(embed_model.encode([question])).astype("float32")
    _, indices = index.search(qv, k=3)
    context = "\n".join([documents[i] for i in indices[0]])
    lang_note = f"Answer fully in {lang_name}." if lang_name != "English" else "Answer fully and completely."
    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile", max_tokens=1024,
        messages=[
            {"role":"system","content":f"You are an expert medical assistant. Give complete, accurate, easy-to-understand answers. Never cut off your response. {lang_note}"},
            {"role":"user","content":f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return res.choices[0].message.content, [documents[i] for i in indices[0]]

dark = st.session_state.dark_mode

if dark:
    bg_main      = "#0a1a12"
    bg_secondary = "#0d2018"
    bg_card      = "rgba(255,255,255,0.04)"
    bg_input     = "rgba(255,255,255,0.07)"
    text_primary = "rgba(255,255,255,0.92)"
    text_secondary = "rgba(255,255,255,0.5)"
    text_muted   = "rgba(255,255,255,0.28)"
    border_color = "rgba(52,211,153,0.2)"
    border_card  = "rgba(255,255,255,0.08)"
    accent1      = "#34d399"
    accent2      = "#10b981"
    accent3      = "#059669"
    btn_gradient = "linear-gradient(135deg,#059669,#10b981)"
    hero_gradient= "linear-gradient(135deg,#34d399,#10b981,#6ee7b7)"
    bubble_user  = "linear-gradient(135deg,#059669,#10b981)"
    bubble_ai_bg = "rgba(52,211,153,0.06)"
    bubble_ai_bd = "rgba(52,211,153,0.15)"
    grid_c       = "rgba(52,211,153,0.03)"
    shadow       = "rgba(52,211,153,0.15)"
else:
    bg_main      = "#f0fdf4"
    bg_secondary = "#dcfce7"
    bg_card      = "white"
    bg_input     = "white"
    text_primary = "#14532d"
    text_secondary = "#166534"
    text_muted   = "#6b7280"
    border_color = "#86efac"
    border_card  = "#bbf7d0"
    accent1      = "#16a34a"
    accent2      = "#15803d"
    accent3      = "#166534"
    btn_gradient = "linear-gradient(135deg,#16a34a,#15803d)"
    hero_gradient= "linear-gradient(135deg,#16a34a,#059669,#047857)"
    bubble_user  = "linear-gradient(135deg,#16a34a,#15803d)"
    bubble_ai_bg = "white"
    bubble_ai_bd = "#bbf7d0"
    grid_c       = "rgba(22,163,74,0.04)"
    shadow       = "rgba(22,163,74,0.12)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

* {{ margin:0; padding:0; box-sizing:border-box; }}

html, body, [data-testid="stAppViewContainer"] {{
    background:{bg_main};
    min-height:100vh;
    font-family:"Plus Jakarta Sans",sans-serif;
}}

[data-testid="stAppViewContainer"]::before {{
    content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
    background:
        radial-gradient(ellipse at 10% 10%, rgba(52,211,153,0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 90%, rgba(16,185,129,0.1) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(110,231,183,0.06) 0%, transparent 60%);
}}

[data-testid="stHeader"] {{ background:transparent !important; }}
[data-testid="stSidebar"] {{ display:none; }}
.block-container {{
    position:relative; z-index:1;
    padding-bottom:180px !important;
    max-width:960px !important;
}}

.grid-bg {{
    position:fixed; inset:0; pointer-events:none; z-index:0;
    background-image:
        linear-gradient({grid_c} 1px, transparent 1px),
        linear-gradient(90deg, {grid_c} 1px, transparent 1px);
    background-size:40px 40px;
    animation:gridMove 25s linear infinite;
}}
@keyframes gridMove {{ 0%{{background-position:0 0}} 100%{{background-position:40px 40px}} }}

.med-bg {{ position:fixed; inset:0; pointer-events:none; z-index:0; overflow:hidden; }}
.med-bg span {{
    position:absolute; font-size:1.4rem;
    opacity:0.04; animation:floatUp 22s linear infinite;
}}
.med-bg span:nth-child(1){{left:5%;animation-delay:0s;animation-duration:18s}}
.med-bg span:nth-child(2){{left:15%;animation-delay:3s;animation-duration:22s}}
.med-bg span:nth-child(3){{left:25%;animation-delay:6s;animation-duration:16s}}
.med-bg span:nth-child(4){{left:35%;animation-delay:1s;animation-duration:24s}}
.med-bg span:nth-child(5){{left:45%;animation-delay:8s;animation-duration:20s}}
.med-bg span:nth-child(6){{left:55%;animation-delay:4s;animation-duration:19s}}
.med-bg span:nth-child(7){{left:65%;animation-delay:2s;animation-duration:23s}}
.med-bg span:nth-child(8){{left:75%;animation-delay:9s;animation-duration:17s}}
.med-bg span:nth-child(9){{left:85%;animation-delay:5s;animation-duration:21s}}
.med-bg span:nth-child(10){{left:92%;animation-delay:7s;animation-duration:25s}}
@keyframes floatUp {{
    0%{{transform:translateY(110vh) rotate(0deg);opacity:0}}
    10%{{opacity:0.04}} 90%{{opacity:0.04}}
    100%{{transform:translateY(-10vh) rotate(360deg);opacity:0}}
}}

/* HERO */
.hero {{ text-align:center; padding:36px 20px 10px; }}
.hero-badge {{
    display:inline-flex; align-items:center; gap:6px;
    background:{bg_card};
    border:1.5px solid {border_color};
    border-radius:50px; padding:6px 16px;
    font-size:0.7rem; color:{accent1};
    letter-spacing:2px; text-transform:uppercase;
    font-weight:700; margin-bottom:14px;
    font-family:"Space Grotesk",sans-serif;
    box-shadow: 0 2px 12px {shadow};
    animation:badgePulse 3s ease infinite;
}}
@keyframes badgePulse {{
    0%,100%{{box-shadow:0 2px 12px {shadow}}}
    50%{{box-shadow:0 4px 24px {shadow},0 0 0 4px rgba(52,211,153,0.1)}}
}}
.hero h1 {{
    font-family:"Space Grotesk",sans-serif;
    font-size:3rem; font-weight:700;
    background:{hero_gradient};
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:8px;
    text-transform:uppercase;
    letter-spacing:2px;
    animation:shimmer 5s linear infinite; background-size:200%;
}}
@keyframes shimmer {{ 0%{{background-position:0%}} 100%{{background-position:200%}} }}
.hero p {{
    color:{text_muted}; font-size:0.95rem; font-weight:400;
    letter-spacing:0.5px;
}}

/* MED CARDS */
.med-cards {{
    display:flex; justify-content:center; gap:10px;
    flex-wrap:wrap; margin:14px auto 18px; max-width:860px;
}}
.med-card {{
    background:{bg_card};
    border:1.5px solid {border_card};
    border-radius:14px; padding:14px 16px;
    text-align:center; min-width:88px;
    backdrop-filter:blur(10px);
    transition:all 0.25s; cursor:pointer;
    animation:cardIn 0.5s ease forwards; opacity:0;
    box-shadow:0 2px 8px {shadow};
}}
.med-card:hover {{
    background:linear-gradient(135deg,rgba(52,211,153,0.12),rgba(16,185,129,0.08));
    border-color:{accent1};
    transform:translateY(-4px);
    box-shadow:0 12px 28px {shadow};
}}
.med-card:active {{ transform:scale(0.96); }}
@keyframes cardIn {{ from{{opacity:0;transform:translateY(12px)}} to{{opacity:1;transform:translateY(0)}} }}
.med-card:nth-child(1){{animation-delay:0.05s}}.med-card:nth-child(2){{animation-delay:0.1s}}
.med-card:nth-child(3){{animation-delay:0.15s}}.med-card:nth-child(4){{animation-delay:0.2s}}
.med-card:nth-child(5){{animation-delay:0.25s}}.med-card:nth-child(6){{animation-delay:0.3s}}
.med-card:nth-child(7){{animation-delay:0.35s}}.med-card:nth-child(8){{animation-delay:0.4s}}
.med-card .icon {{ font-size:1.7rem; margin-bottom:6px; }}
.med-card .label {{
    font-size:0.58rem; color:{text_muted};
    text-transform:uppercase; letter-spacing:1px;
    font-family:"Space Grotesk",sans-serif; font-weight:600;
}}

/* STATS */
.stats-bar {{
    display:flex; justify-content:center; gap:32px;
    padding:12px 0 16px;
    border-top:1px solid {border_card};
    border-bottom:1px solid {border_card};
    margin:0 0 16px;
    background:{bg_card};
    border-radius:14px;
    box-shadow:0 2px 8px {shadow};
}}
.stat {{ text-align:center; }}
.stat-num {{
    font-family:"Space Grotesk",sans-serif;
    font-size:1.2rem; font-weight:700; color:{accent1};
}}
.stat-lbl {{
    font-size:0.62rem; color:{text_muted};
    text-transform:uppercase; letter-spacing:1px; font-weight:600;
}}

/* MESSAGES */
.msg-user {{ display:flex; justify-content:flex-end; margin:12px 0; }}
.msg-user .bubble {{
    background:{bubble_user}; color:white;
    padding:12px 18px; border-radius:18px 18px 4px 18px; max-width:68%;
    font-size:0.92rem; line-height:1.65;
    box-shadow:0 4px 16px {shadow};
    animation:msgIn 0.3s ease;
    font-family:"Plus Jakarta Sans",sans-serif;
}}
.msg-ai {{ display:flex; justify-content:flex-start; margin:12px 0; gap:10px; align-items:flex-start; }}
.ai-icon {{
    width:36px; height:36px; border-radius:50%;
    background:{btn_gradient};
    display:flex; align-items:center; justify-content:center;
    font-size:0.95rem; flex-shrink:0; margin-top:2px;
    box-shadow:0 2px 10px {shadow};
}}
.msg-ai .bubble {{
    background:{bubble_ai_bg};
    border:1.5px solid {bubble_ai_bd};
    color:{text_primary}; padding:12px 18px;
    border-radius:4px 18px 18px 18px; max-width:68%;
    font-size:0.92rem; line-height:1.8;
    backdrop-filter:blur(10px);
    animation:msgIn 0.3s ease;
    font-family:"Plus Jakarta Sans",sans-serif;
    box-shadow:0 2px 12px {shadow};
}}
@keyframes msgIn {{ from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:translateY(0)}} }}

.sources-wrap {{ margin-left:46px; margin-top:4px; max-width:68%; margin-bottom:8px; }}
.sources-wrap details {{
    background:{bg_card}; border:1.5px solid {border_card};
    border-radius:10px; padding:7px 14px; cursor:pointer;
    box-shadow:0 1px 4px {shadow};
}}
.sources-wrap summary {{
    color:{text_muted}; font-size:0.7rem;
    text-transform:uppercase; letter-spacing:1px;
    list-style:none; font-weight:600;
    font-family:"Space Grotesk",sans-serif;
}}
.sources-wrap span {{
    display:block; color:{text_secondary}; font-size:0.78rem;
    padding:5px 0; border-bottom:1px solid {border_card}; line-height:1.5;
}}
.sources-wrap span:last-child {{ border-bottom:none; }}

.typing-dot {{
    display:inline-block; width:7px; height:7px; border-radius:50%;
    background:{accent1}; margin:0 2px; animation:tdot 1.2s infinite;
}}
.typing-dot:nth-child(2){{animation-delay:0.2s}} .typing-dot:nth-child(3){{animation-delay:0.4s}}
@keyframes tdot {{
    0%,80%,100%{{transform:translateY(0);opacity:0.35}}
    40%{{transform:translateY(-6px);opacity:1}}
}}

/* INPUTS */
.stTextInput input {{
    background:{bg_input} !important;
    border:1.5px solid {border_color} !important;
    border-radius:50px !important; color:{text_primary} !important;
    font-family:"Plus Jakarta Sans",sans-serif !important;
    font-size:0.92rem !important; padding:14px 22px !important;
    box-shadow:0 2px 8px {shadow} !important;
}}
.stTextInput input:focus {{
    border-color:{accent1} !important;
    box-shadow:0 0 0 3px rgba(52,211,153,0.15) !important;
}}
.stTextInput input::placeholder {{ color:{text_muted} !important; }}

.stButton > button {{
    background:{btn_gradient} !important; color:white !important;
    border:none !important; border-radius:50px !important;
    padding:12px 20px !important;
    font-family:"Space Grotesk",sans-serif !important;
    font-weight:700 !important; font-size:0.84rem !important;
    white-space:nowrap !important; width:100% !important;
    letter-spacing:0.5px !important; text-transform:uppercase !important;
    box-shadow:0 4px 14px {shadow} !important;
    transition:all 0.2s !important;
}}
.stButton > button:hover {{
    box-shadow:0 6px 22px {shadow} !important;
    transform:translateY(-2px) !important;
    filter:brightness(1.08) !important;
}}

.stSelectbox > div > div {{
    background:{bg_input} !important;
    border:1.5px solid {border_color} !important;
    border-radius:50px !important; color:{text_primary} !important;
    font-size:0.83rem !important;
    box-shadow:0 2px 8px {shadow} !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    background:{bg_card} !important;
    border-radius:50px !important; padding:4px !important;
    gap:4px !important; border:1.5px solid {border_color} !important;
    box-shadow:0 2px 10px {shadow} !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius:50px !important; color:{text_muted} !important;
    font-family:"Space Grotesk",sans-serif !important;
    font-size:0.8rem !important; padding:8px 18px !important;
    font-weight:600 !important; text-transform:uppercase !important;
    letter-spacing:0.5px !important;
}}
.stTabs [aria-selected="true"] {{
    background:{btn_gradient} !important; color:white !important;
    box-shadow:0 2px 10px {shadow} !important;
}}

.stTextArea textarea {{
    background:{bg_input} !important;
    border:1.5px solid {border_color} !important;
    border-radius:16px !important; color:{text_primary} !important;
    font-family:"Plus Jakarta Sans",sans-serif !important;
    font-size:0.88rem !important;
    box-shadow:0 2px 8px {shadow} !important;
}}

.stNumberInput input {{
    background:{bg_input} !important;
    border:1.5px solid {border_color} !important;
    border-radius:12px !important; color:{text_primary} !important;
    box-shadow:0 2px 8px {shadow} !important;
}}

.section-card {{
    background:{bg_card};
    border:1.5px solid {border_card};
    border-radius:16px; padding:20px 24px; margin-bottom:16px;
    box-shadow:0 4px 16px {shadow};
}}
.section-title {{
    color:{accent1}; font-family:"Space Grotesk",sans-serif;
    font-size:1rem; font-weight:700; margin-bottom:6px;
    text-transform:uppercase; letter-spacing:1px;
}}
.section-sub {{
    color:{text_muted}; font-size:0.84rem;
}}

.result-box {{
    background:{bubble_ai_bg}; border:1.5px solid {bubble_ai_bd};
    border-radius:14px; padding:18px 22px; color:{text_primary};
    font-size:0.91rem; line-height:1.8; margin-top:12px;
    box-shadow:0 4px 16px {shadow};
}}

.emergency-card {{
    background:{bg_card}; border:1.5px solid rgba(239,68,68,0.3);
    border-radius:12px; padding:14px 18px; margin:6px 0;
    display:flex; align-items:center; gap:14px;
    box-shadow:0 2px 8px rgba(239,68,68,0.08);
    transition:all 0.2s;
}}
.emergency-card:hover {{ transform:translateY(-1px); box-shadow:0 4px 14px rgba(239,68,68,0.15); }}
.emergency-card .num {{
    font-family:"Space Grotesk",sans-serif;
    font-size:1.1rem; font-weight:700; color:#ef4444;
}}
.emergency-card .elabel {{ font-size:0.8rem; color:{text_muted}; font-weight:500; }}

.action-row {{ display:flex; gap:8px; margin-left:46px; margin-top:6px; margin-bottom:4px; align-items:center; }}
.action-btn {{
    background:{bg_card}; border:1.5px solid {border_card};
    border-radius:50px; color:{text_muted};
    font-size:0.72rem; padding:4px 12px; cursor:pointer;
    transition:all 0.2s; display:inline-flex; align-items:center; gap:4px;
    font-family:"Space Grotesk",sans-serif; font-weight:600;
    box-shadow:0 1px 4px {shadow};
}}
.action-btn:hover {{
    background:rgba(52,211,153,0.1); border-color:{accent1}; color:{accent1};
}}

.watermark {{
    position:fixed; bottom:10px; right:16px; font-size:0.6rem;
    color:{text_muted}; letter-spacing:2px; text-transform:uppercase;
    pointer-events:none; z-index:0; opacity:0.4;
    font-family:"Space Grotesk",sans-serif;
}}
</style>

<div class="grid-bg"></div>
<div class="med-bg">
  <span>🩺</span><span>💊</span><span>🧬</span><span>🏥</span><span>❤️</span>
  <span>🩻</span><span>💉</span><span>🔬</span><span>🩹</span><span>⚕️</span>
</div>
<div class="watermark">MediRAG AI • Powered by LLaMA 3</div>

<div id="medPopup" style="display:none;position:fixed;inset:0;z-index:9999;
    background:rgba(0,0,0,0.6);backdrop-filter:blur(8px);align-items:center;justify-content:center;">
  <div style="background:{bg_card};border:2px solid {border_color};
      border-radius:22px;padding:32px;max-width:420px;width:90%;
      box-shadow:0 24px 60px {shadow};">
    <div style="font-size:2.8rem;text-align:center;margin-bottom:12px;" id="popIcon">🩺</div>
    <h3 style="font-family:Space Grotesk,sans-serif;color:{text_primary};font-size:1.3rem;
        text-align:center;margin-bottom:14px;text-transform:uppercase;letter-spacing:1px;" id="popTitle"></h3>
    <p style="color:{text_secondary};font-size:0.87rem;line-height:1.85;margin-bottom:22px;
        font-family:Plus Jakarta Sans,sans-serif;" id="popDesc"></p>
    <div style="display:flex;gap:10px;justify-content:center;">
      <button onclick="askCard()" style="background:{btn_gradient};color:white;border:none;
          border-radius:50px;padding:11px 24px;font-size:0.83rem;font-weight:700;cursor:pointer;
          font-family:Space Grotesk,sans-serif;text-transform:uppercase;letter-spacing:0.5px;
          box-shadow:0 4px 14px {shadow};">ASK ABOUT THIS →</button>
      <button onclick="closeCard()" style="background:{bg_card};color:{text_muted};
          border:1.5px solid {border_color};border-radius:50px;
          padding:11px 24px;font-size:0.83rem;cursor:pointer;
          font-family:Space Grotesk,sans-serif;font-weight:600;">CLOSE</button>
    </div>
  </div>
</div>

<script>
var curCard="";
var cardInfo={{
  "Diagnosis":{{icon:"🩺",desc:"Medical diagnosis identifies diseases through symptoms, tests and examinations. Early and accurate diagnosis is critical for effective treatment. Doctors use blood tests, imaging scans, and physical exams. It forms the foundation of all medical care and treatment planning."}},
  "Medicines":{{icon:"💊",desc:"Medicines treat, prevent or manage diseases and health conditions. They must be taken at the correct dosage and timing for best results. Some medicines require a prescription while others are available over the counter. Always consult a doctor before starting or stopping any medication."}},
  "Genetics":{{icon:"🧬",desc:"Genetics studies genes and hereditary traits passed from parents to children. Many diseases have genetic components that increase risk in families. Genetic testing can identify inherited conditions early before symptoms appear. Gene therapy is an emerging treatment that corrects faulty genes directly."}},
  "Hospital":{{icon:"🏥",desc:"Hospitals provide comprehensive medical care from emergency to specialized surgery. They have advanced equipment for diagnosis, monitoring and treatment. Different departments handle specific conditions like cardiology and neurology. Timely hospital care significantly improves survival and recovery rates."}},
  "Research":{{icon:"🔬",desc:"Medical research discovers new treatments, drugs and healthcare approaches. Clinical trials test new therapies rigorously before approving for public use. Research has led to vaccines, antibiotics and life-saving surgical techniques. Evidence-based medicine relies on solid research for all treatment decisions."}},
  "Radiology":{{icon:"🩻",desc:"Radiology uses imaging technology to see inside the human body. X-rays, MRI, CT scans and ultrasound are common radiology tools. Radiologists detect tumors, fractures, infections and organ abnormalities. Modern radiology is minimally invasive and provides incredibly detailed internal views."}},
  "Cardiology":{{icon:"❤️",desc:"Cardiology specializes in diagnosing and treating heart and blood vessel diseases. Common conditions include heart attacks, hypertension and irregular heartbeats. Cardiologists use ECG, echo and stress tests to assess heart health. A healthy lifestyle with exercise and diet prevents most cardiovascular diseases."}},
  "Neurology":{{icon:"🧠",desc:"Neurology deals with diseases of the brain, spinal cord and nervous system. Stroke, epilepsy, Alzheimer's and Parkinson's are common neurological disorders. Diagnosis uses MRI, EEG and nerve conduction studies for accuracy. Early treatment is crucial as brain damage can be rapid and irreversible."}},
}};
function openCard(n){{
    curCard=n; var d=cardInfo[n];
    document.getElementById("popIcon").innerText=d.icon;
    document.getElementById("popTitle").innerText=n.toUpperCase();
    document.getElementById("popDesc").innerText=d.desc;
    document.getElementById("medPopup").style.display="flex";
}}
function closeCard(){{ document.getElementById("medPopup").style.display="none"; }}
function askCard(){{
    closeCard();
    var ins=window.parent.document.querySelectorAll("input[type=text]");
    ins.forEach(function(i){{
        var s=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,"value").set;
        s.call(i,"Tell me about "+curCard+" in medicine");
        i.dispatchEvent(new Event("input",{{bubbles:true}}));
    }});
}}
function copyText(text){{
    navigator.clipboard.writeText(text).then(function(){{
        alert("Copied to clipboard!");
    }});
}}
document.getElementById("medPopup").onclick=function(e){{if(e.target===this)closeCard();}};
</script>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <div class="hero-badge">🩺 AI MEDICAL ASSISTANT</div>
  <h1>MEDIRAG AI</h1>
  <p>Ask any health question — instant, accurate answers in your language</p>
</div>
<div class="med-cards">
  <div class="med-card" onclick="openCard('Diagnosis')"><div class="icon">🩺</div><div class="label">Diagnosis</div></div>
  <div class="med-card" onclick="openCard('Medicines')"><div class="icon">💊</div><div class="label">Medicines</div></div>
  <div class="med-card" onclick="openCard('Genetics')"><div class="icon">🧬</div><div class="label">Genetics</div></div>
  <div class="med-card" onclick="openCard('Hospital')"><div class="icon">🏥</div><div class="label">Hospital</div></div>
  <div class="med-card" onclick="openCard('Research')"><div class="icon">🔬</div><div class="label">Research</div></div>
  <div class="med-card" onclick="openCard('Radiology')"><div class="icon">🩻</div><div class="label">Radiology</div></div>
  <div class="med-card" onclick="openCard('Cardiology')"><div class="icon">❤️</div><div class="label">Cardiology</div></div>
  <div class="med-card" onclick="openCard('Neurology')"><div class="icon">🧠</div><div class="label">Neurology</div></div>
</div>
<div class="stats-bar">
  <div class="stat"><div class="stat-num">15+</div><div class="stat-lbl">Medical Topics</div></div>
  <div class="stat"><div class="stat-num">10</div><div class="stat-lbl">Languages</div></div>
  <div class="stat"><div class="stat-num">RAG</div><div class="stat-lbl">AI Powered</div></div>
  <div class="stat"><div class="stat-num">LLaMA 3</div><div class="stat-lbl">Model</div></div>
</div>
""", unsafe_allow_html=True)

h1, h2, h3, h4 = st.columns([3,1,1,1])
with h2:
    sel = st.selectbox("🌐", list(LANGUAGES.keys()), key="lang_sel", label_visibility="collapsed")
    st.session_state.lang = sel
with h3:
    dm = "🌙 DARK" if not dark else "☀️ LIGHT"
    if st.button(dm, key="mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
with h4:
    if st.button("🗑️ CLEAR", key="clear"):
        st.session_state.messages = []
        save_history()
        st.rerun()

tab_main, tab_symptom, tab_bmi, tab_medicine, tab_emergency = st.tabs([
    "💬 ASK AI", "🔍 SYMPTOMS", "⚖️ BMI", "💊 MEDICINE", "🚨 EMERGENCY"
])

with tab_symptom:
    st.markdown(f'<div class="section-card"><div class="section-title">🔍 AI Symptom Checker</div><div class="section-sub">Enter your symptoms and AI will suggest possible conditions</div></div>', unsafe_allow_html=True)
    sym_in = st.text_area("YOUR SYMPTOMS:", placeholder="e.g. fever, headache, body pain, fatigue", height=100, key="sym_in")
    if st.button("🔍 ANALYZE SYMPTOMS", key="sym_btn"):
        if sym_in.strip():
            with st.spinner("Analyzing symptoms..."):
                r = client.chat.completions.create(
                    model="llama-3.3-70b-versatile", max_tokens=800,
                    messages=[
                        {"role":"system","content":"You are an expert medical assistant. Given symptoms, list top 3-5 possible conditions with brief explanation. Add disclaimer to consult a doctor."},
                        {"role":"user","content":f"Symptoms: {sym_in}\n\nWhat are the possible conditions?"}
                    ]
                )
                st.markdown(f'<div class="result-box">{r.choices[0].message.content.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter at least one symptom.")

with tab_bmi:
    st.markdown(f'<div class="section-card"><div class="section-title">⚖️ BMI Calculator</div><div class="section-sub">Calculate your Body Mass Index and get personalized AI health advice</div></div>', unsafe_allow_html=True)
    bc1, bc2 = st.columns(2)
    with bc1:
        weight = st.number_input("WEIGHT (KG)", min_value=1.0, max_value=300.0, value=70.0, step=0.5)
    with bc2:
        height_cm = st.number_input("HEIGHT (CM)", min_value=50.0, max_value=250.0, value=170.0, step=0.5)
    age = st.slider("AGE", 1, 100, 25)
    gender = st.radio("GENDER", ["Male","Female"], horizontal=True)
    if st.button("⚖️ CALCULATE BMI", key="bmi_btn"):
        bmi = weight / ((height_cm/100)**2)
        if bmi < 18.5: cat,col,tip = "UNDERWEIGHT","#3b82f6","Increase nutrient-rich foods and healthy calorie intake."
        elif bmi < 25: cat,col,tip = "NORMAL WEIGHT","#10b981","Keep it up with balanced diet and regular exercise!"
        elif bmi < 30: cat,col,tip = "OVERWEIGHT","#f59e0b","Try moderate exercise and reduce processed foods."
        else: cat,col,tip = "OBESE","#ef4444","Please consult a doctor and focus on diet and physical activity."
        st.markdown(f"""<div style='background:{bg_card};border:2px solid {col}33;
        border-radius:18px;padding:28px;text-align:center;margin-top:10px;box-shadow:0 6px 24px {shadow};'>
            <div style='font-size:3.5rem;font-family:Space Grotesk,sans-serif;font-weight:800;color:{col};'>{bmi:.1f}</div>
            <div style='font-size:1rem;color:{col};font-weight:700;margin:8px 0;letter-spacing:2px;'>{cat}</div>
            <div style='color:{text_muted};font-size:0.85rem;margin-top:6px;'>{tip}</div>
            <div style='margin-top:18px;display:flex;justify-content:center;gap:28px;'>
                <div><div style='color:{text_muted};font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;'>Weight</div><div style='color:{text_primary};font-weight:700;font-size:0.95rem;'>{weight}kg</div></div>
                <div><div style='color:{text_muted};font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;'>Height</div><div style='color:{text_primary};font-weight:700;font-size:0.95rem;'>{height_cm}cm</div></div>
                <div><div style='color:{text_muted};font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;'>Age</div><div style='color:{text_primary};font-weight:700;font-size:0.95rem;'>{age}yrs</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
        with st.spinner("Getting personalized tips..."):
            ba = client.chat.completions.create(
                model="llama-3.3-70b-versatile", max_tokens=300,
                messages=[
                    {"role":"system","content":"You are a health advisor. Give 3 brief practical tips."},
                    {"role":"user","content":f"Age {age}, {gender}, BMI {bmi:.1f} ({cat}). Give 3 health tips."}
                ]
            )
            st.markdown(f'<div style="background:rgba(52,211,153,0.08);border:1.5px solid {border_color};border-radius:14px;padding:16px 20px;color:{text_primary};font-size:0.88rem;line-height:1.75;margin-top:14px;box-shadow:0 2px 10px {shadow};"> 💡 {ba.choices[0].message.content.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)

with tab_medicine:
    st.markdown(f'<div class="section-card"><div class="section-title">💊 Medicine Information</div><div class="section-sub">Enter a medicine name to get dosage, uses and side effects</div></div>', unsafe_allow_html=True)
    med_name = st.text_input("MEDICINE NAME:", placeholder="e.g. Paracetamol, Metformin, Aspirin", key="med_name")
    if st.button("💊 GET MEDICINE INFO", key="med_btn"):
        if med_name.strip():
            with st.spinner("Looking up medicine information..."):
                mr = client.chat.completions.create(
                    model="llama-3.3-70b-versatile", max_tokens=600,
                    messages=[
                        {"role":"system","content":"You are a pharmacist. Give clear medicine information with sections: Uses, Standard Dosage, Side Effects, Precautions. Add disclaimer to consult a doctor."},
                        {"role":"user","content":f"Give information about: {med_name}"}
                    ]
                )
                st.markdown(f'<div class="result-box">{mr.choices[0].message.content.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a medicine name.")

with tab_emergency:
    st.markdown(f"""<div style="background:rgba(239,68,68,0.06);border:1.5px solid rgba(239,68,68,0.25);border-radius:16px;padding:20px 24px;margin-bottom:16px;box-shadow:0 4px 16px rgba(239,68,68,0.08);"><div style="color:#ef4444;font-family:Space Grotesk,sans-serif;font-size:1rem;font-weight:700;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px;">🚨 Emergency Contacts — India</div><div style="color:{text_muted};font-size:0.84rem;">Save these numbers. In an emergency, call immediately.</div></div>""", unsafe_allow_html=True)
    emergencies = [
        ("🚑","108","Ambulance (National)"),("🚒","101","Fire & Rescue"),
        ("👮","100","Police"),("📞","112","All Emergencies"),
        ("🏥","104","Health Helpline"),("👩","181","Women Helpline"),
        ("🧒","1098","Child Helpline"),("🧠","9152987821","Mental Health — iCall"),
        ("❤️","1800-112-545","Cardiac Emergency"),("🐍","1800-425-1213","Poison Control"),
    ]
    ec1, ec2 = st.columns(2)
    for i,(icon,num,label) in enumerate(emergencies):
        with (ec1 if i%2==0 else ec2):
            st.markdown(f'<div class="emergency-card"><div style="font-size:1.8rem;">{icon}</div><div><div class="num">{num}</div><div class="elabel">{label}</div></div></div>', unsafe_allow_html=True)
    st.markdown(f"""<div style='background:{bg_card};border:1.5px solid {border_card};border-radius:16px;
    padding:20px 24px;margin-top:16px;box-shadow:0 4px 16px {shadow};'>
    <div style='color:{accent1};font-family:Space Grotesk,sans-serif;font-size:0.9rem;font-weight:700;
    margin-bottom:14px;text-transform:uppercase;letter-spacing:1px;'>🆘 First Aid Quick Guide</div>""", unsafe_allow_html=True)
    for icon,title,tip in [
        ("🫀","HEART ATTACK","Call 108 immediately. Have person sit or lie down. Loosen tight clothing. Give aspirin if available."),
        ("🌡️","HIGH FEVER","Give paracetamol. Apply cool cloth on forehead. Stay hydrated. See doctor if above 103°F."),
        ("🩸","HEAVY BLEEDING","Apply firm pressure with clean cloth. Elevate injured area. Do not remove cloth if soaked."),
        ("🔥","BURNS","Cool under running water for 10 mins. No ice or butter. Cover with clean cloth. Seek medical help."),
    ]:
        st.markdown(f'<div style="padding:12px 0;border-bottom:1px solid {border_card};"><div style="color:{text_primary};font-weight:700;font-size:0.85rem;margin-bottom:4px;font-family:Space Grotesk,sans-serif;">{icon} {title}</div><div style="color:{text_muted};font-size:0.82rem;line-height:1.65;">{tip}</div></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_main:
    if not st.session_state.messages:
        st.markdown(f"<p style='text-align:center;color:{text_muted};font-size:0.74rem;margin-bottom:10px;font-family:Space Grotesk,sans-serif;font-weight:600;letter-spacing:1px;text-transform:uppercase;'>Quick questions — click to ask</p>", unsafe_allow_html=True)
        suggestions = [
            ("Symptoms of diabetes?",  "linear-gradient(135deg,#16a34a,#15803d)"),
            ("How malaria spreads?",   "linear-gradient(135deg,#0369a1,#0284c7)"),
            ("Kidney stone causes?",   "linear-gradient(135deg,#dc2626,#b91c1c)"),
            ("What is Parkinson\'s?", "linear-gradient(135deg,#7c3aed,#6d28d9)"),
            ("Treat high BP?",         "linear-gradient(135deg,#0d9488,#0f766e)"),
            ("Foods rich in iron?",    "linear-gradient(135deg,#b45309,#92400e)"),
            ("COVID-19 symptoms?",     "linear-gradient(135deg,#be185d,#9d174d)"),
            ("How to treat fever?",    "linear-gradient(135deg,#0891b2,#0e7490)"),
        ]
        r1 = st.columns(4)
        r2 = st.columns(4)
        for i,(q,g) in enumerate(suggestions):
            with (r1[i] if i<4 else r2[i-4]):
                if st.button(q, key=f"sq{i}"):
                    st.session_state.pending = q
                    st.rerun()

    for idx_m, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user"><div class="bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-ai"><div class="ai-icon">🩺</div><div class="bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
            escaped = msg["content"].replace("'","\'").replace(chr(10),"\\n")
            fb = st.session_state.feedback.get(str(idx_m),"")
            fa, fb_col, fc = st.columns([1,1,8])
            with fa:
                if st.button("👍", key=f"like_{idx_m}"):
                    st.session_state.feedback[str(idx_m)] = "like"
                    st.rerun()
            with fb_col:
                if st.button("👎", key=f"dis_{idx_m}"):
                    st.session_state.feedback[str(idx_m)] = "dislike"
                    st.rerun()
            with fc:
                fb_status = st.session_state.feedback.get(str(idx_m),"")
                fb_tag = f"<span style='color:#10b981;font-size:0.72rem;font-weight:600;'>✓ Helpful</span>" if fb_status=="like" else f"<span style='color:#ef4444;font-size:0.72rem;font-weight:600;'>✗ Not helpful</span>" if fb_status=="dislike" else ""
                st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;padding:5px 0;"><button class="action-btn" onclick="copyText('{escaped}')">📋 COPY</button>{fb_tag}</div>""", unsafe_allow_html=True)
            if "sources" in msg:
                src = "".join([f"<span>📄 {s[:120]}...</span>" for s in msg["sources"]])
                st.markdown(f'<div class="sources-wrap"><details><summary>📚 References</summary>{src}</details></div>', unsafe_allow_html=True)

    lang_code = LANGUAGES[st.session_state.lang]
    st.components.v1.html(f"""
    <style>
    .mo{{width:46px;height:46px;border-radius:50%;position:relative;flex-shrink:0;
        background:rgba(16,185,129,0.12);border:2px solid rgba(16,185,129,0.4);
        display:flex;align-items:center;justify-content:center;cursor:pointer;
        transition:all 0.2s;font-size:1.1rem;color:white;
        box-shadow:0 2px 10px rgba(16,185,129,0.15);}}
    .mo:hover{{background:rgba(16,185,129,0.2);transform:scale(1.06)}}
    .mo.on{{background:rgba(239,68,68,0.2);border-color:rgba(239,68,68,0.5)}}
    .mr{{position:absolute;inset:-7px;border-radius:50%;border:2px solid rgba(239,68,68,0.4);opacity:0}}
    .mo.on .mr{{animation:rp 1s infinite;opacity:1}}
    @keyframes rp{{0%{{transform:scale(1);opacity:0.6}}100%{{transform:scale(1.65);opacity:0}}}}
    .ms{{color:rgba(100,130,110,0.8);font-size:0.72rem;font-family:Space Grotesk,sans-serif;font-weight:500;}}
    </style>
    <div style="display:flex;align-items:center;gap:10px;padding:6px 0 8px;">
      <div class="mo" id="mb" onclick="tv()"><div class="mr"></div>🎤</div>
      <span class="ms" id="mst">CLICK 🎤 TO SPEAK IN {st.session_state.lang.upper()}</span>
    </div>
    <script>
    var on=false,r=null;
    function tv(){{
        if(!window.SpeechRecognition&&!window.webkitSpeechRecognition){{
            document.getElementById("mst").innerText="USE CHROME FOR VOICE INPUT";return;
        }}
        if(on){{r.stop();return;}}
        r=new(window.SpeechRecognition||window.webkitSpeechRecognition)();
        r.lang="{lang_code}";r.interimResults=false;on=true;
        var b=document.getElementById("mb");
        b.classList.add("on");b.innerHTML='<div class="mr"></div>🔴';
        document.getElementById("mst").innerText="LISTENING... SPEAK NOW";
        r.start();
        r.onresult=function(e){{
            var t=e.results[0][0].transcript;
            var ins=window.parent.document.querySelectorAll("input[type=text]");
            ins.forEach(function(i){{
                var s=Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype,"value").set;
                s.call(i,t);i.dispatchEvent(new Event("input",{{bubbles:true}}));
            }});
            document.getElementById("mst").innerText="✅ "+t.toUpperCase();rst();
        }};
        r.onerror=function(e){{document.getElementById("mst").innerText="❌ "+e.error.toUpperCase();rst();}};
        r.onend=function(){{rst();}};
    }}
    function rst(){{
        on=false;var b=document.getElementById("mb");
        b.classList.remove("on");b.innerHTML='<div class="mr"></div>🎤';
    }}
    </script>
    """, height=62)

    c1c, c2c = st.columns([6,1])
    with c1c:
        user_input = st.text_input("q",
            placeholder="Type your medical question here...",
            label_visibility="collapsed",
            key=f"inp_{st.session_state.ikey}")
    with c2c:
        send = st.button("SEND →", key="send")

    final_q = st.session_state.pending or (user_input if send else "")
    if final_q:
        st.session_state.pending = ""
        st.session_state.ikey += 1
        st.session_state.messages.append({"role":"user","content":final_q})
        ph = st.empty()
        ph.markdown(f'<div class="msg-ai"><div class="ai-icon">🩺</div><div class="bubble"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div></div>', unsafe_allow_html=True)
        answer, sources = ask(final_q, st.session_state.lang)
        time.sleep(0.3)
        ph.empty()
        st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources})
        save_history()
        st.rerun()
