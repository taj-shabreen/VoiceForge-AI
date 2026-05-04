"""
main.py  –  VoiceForge AI · Multi-Sample Voice Cloning System
Run:  streamlit run app/main.py
"""

import os, time, sys, warnings
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ── Absolute paths (fixes all relative-path errors) ───────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR   = os.path.join(BASE_DIR, "temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(TEMP_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from audio_processor import clean_audio, merge_audio_files, analyze_audio, get_waveform_data
from tts_engine import generate_speech, SUPPORTED_LANGUAGES

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="VoiceForge AI", page_icon="🎙️",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root {
    --bg:#08090d; --surface:#111318; --surface2:#181c24; --border:#252a35;
    --accent:#5c6ef8; --accent2:#a78bfa; --success:#22d3a5;
    --warn:#f59e0b; --danger:#f43f5e; --text:#e8eaf0; --muted:#6b7280;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text);font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
#MainMenu,footer,header{visibility:hidden;}

.hero{background:linear-gradient(135deg,#0d0f1a 0%,#1a1040 50%,#0a1628 100%);border:1px solid var(--border);border-radius:20px;padding:48px 40px 40px;margin-bottom:32px;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-80px;right:-80px;width:300px;height:300px;background:radial-gradient(circle,rgba(92,110,248,.18) 0%,transparent 70%);pointer-events:none;}
.hero-title{font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;background:linear-gradient(90deg,#a78bfa,#5c6ef8,#22d3a5);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;margin:0 0 10px;}
.hero-sub{color:var(--muted);font-size:1.05rem;font-weight:300;}

.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:24px;margin-bottom:20px;}
.card-title{font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;color:var(--accent2);letter-spacing:.5px;margin-bottom:16px;display:flex;align-items:center;gap:8px;}
.step-label{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;font-family:'Syne',sans-serif;font-weight:700;font-size:.85rem;flex-shrink:0;}

.badge{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:999px;font-size:.78rem;font-weight:500;}
.badge-success{background:rgba(34,211,165,.12);color:var(--success);border:1px solid rgba(34,211,165,.25);}
.badge-warn{background:rgba(245,158,11,.12);color:var(--warn);border:1px solid rgba(245,158,11,.25);}
.badge-danger{background:rgba(244,63,94,.12);color:var(--danger);border:1px solid rgba(244,63,94,.25);}
.badge-info{background:rgba(92,110,248,.12);color:var(--accent);border:1px solid rgba(92,110,248,.25);}

.metrics-row{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px;}
.metric-chip{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:10px 16px;min-width:100px;text-align:center;}
.metric-chip .val{font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:700;color:var(--accent);}
.metric-chip .lbl{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-top:2px;}

/* recorder box */
.rec-box{background:var(--surface2);border:2px dashed var(--border);border-radius:14px;padding:20px;text-align:center;margin:12px 0;}
.rec-tip{font-size:.82rem;color:var(--muted);margin-top:8px;line-height:1.5;}

.stTextArea textarea,.stTextInput input{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text)!important;font-family:'DM Sans',sans-serif!important;}
.stTextArea textarea:focus,.stTextInput input:focus{border-color:var(--accent)!important;box-shadow:0 0 0 2px rgba(92,110,248,.2)!important;}
.stSelectbox>div>div{background:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:10px!important;}
[data-testid="stFileUploader"]{background:var(--surface2)!important;border:1.5px dashed var(--border)!important;border-radius:12px!important;padding:12px!important;}
.stButton>button{width:100%;background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:#fff!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:1.05rem!important;border:none!important;border-radius:12px!important;padding:14px 0!important;}
.stButton>button:hover{opacity:.88!important;}
audio{width:100%;border-radius:10px;}
hr{border-color:var(--border)!important;margin:28px 0;}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] .stMarkdown{color:var(--text)!important;font-family:'DM Sans',sans-serif;}
.sidebar-section{background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:16px;}
.sidebar-section h4{font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;color:var(--accent2);text-transform:uppercase;letter-spacing:1px;margin:0 0 12px;}
.history-item{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:12px 16px;margin-bottom:8px;font-size:.88rem;}
.history-item .hist-text{color:var(--text);font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:260px;}
.history-item .hist-meta{color:var(--muted);font-size:.75rem;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:       st.session_state.history = []
if "generation_count" not in st.session_state: st.session_state.generation_count = 0

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 20px;">
      <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                  background:linear-gradient(90deg,#a78bfa,#5c6ef8);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        🎙️ VoiceForge
      </div>
      <div style="font-size:.72rem;color:#6b7280;letter-spacing:1.5px;text-transform:uppercase;margin-top:4px;">
        AI Voice Cloning
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><h4>⚙️ Generation Settings</h4>', unsafe_allow_html=True)
    language = st.selectbox("Language", list(SUPPORTED_LANGUAGES.keys()), index=0)
    speed    = st.slider("Speech Speed", 0.5, 1.8, 1.0, 0.05, help="1.0 = natural")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section"><h4>📋 History</h4>', unsafe_allow_html=True)
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"""<div class="history-item">
                <div class="hist-text">"{item['text'][:50]}{'…' if len(item['text'])>50 else ''}"</div>
                <div class="hist-meta">🌐 {item['lang']} · ⏱ {item['gen_time']}s · 🔊 {item['dur']}s</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#6b7280;font-size:.85rem;text-align:center;padding:10px 0;">No generations yet</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="sidebar-section">
        <h4>📊 Session Stats</h4>
        <div style="display:flex;justify-content:space-between;color:#e8eaf0;font-size:.9rem;">
            <span>Total Generated</span>
            <span style="font-family:'Syne',sans-serif;font-weight:700;color:#5c6ef8;">{st.session_state.generation_count}</span>
        </div></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sidebar-section">
        <h4>🔀 Pipeline</h4>
        <div style="font-size:.78rem;color:#9ca3af;line-height:2;">
        🎤 Voice Input<br>↓ Clean & Normalize<br>↓ Speaker Embedding<br>↓ XTTS v2 Inference<br>↓ 🔊 Audio Output
        </div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-title">VoiceForge AI</div>
  <div class="hero-sub">Deep learning voice cloning · Powered by XTTS v2 · 16 languages · Real-time generation</div>
  <div style="margin-top:18px;display:flex;gap:10px;flex-wrap:wrap;">
    <span class="badge badge-success">● XTTS v2 Active</span>
    <span class="badge badge-info">● 16 Languages</span>
    <span class="badge badge-info">● Built-in Recorder</span>
    <span class="badge badge-warn">● GPU Accelerated</span>
  </div>
</div>""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 0.9], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
#  LEFT PANEL
# ══════════════════════════════════════════════════════════════════════════════
with col_left:

    # ── Step 1: Text ──────────────────────────────────────────────────────────
    st.markdown("""<div class="card">
        <div class="card-title"><span class="step-label">1</span> Text to Synthesize</div>""",
        unsafe_allow_html=True)

    text_input = st.text_area(
        label="Text to synthesize",
        label_visibility="collapsed",
        placeholder="Type any sentence — it will be spoken in the cloned voice…",
        height=130, key="text_input")

    wc = len(text_input.split()) if text_input.strip() else 0
    cc = len(text_input)
    q  = "badge-success" if 10<=wc<=200 else "badge-warn" if wc>0 else "badge-info"
    ql = "✓ Good length" if 10<=wc<=200 else "↑ Add more text" if 0<wc<10 else "↓ Try shorter" if wc>200 else "Enter text above"
    st.markdown(f"""<div style="display:flex;justify-content:space-between;margin-top:-8px;margin-bottom:4px;">
        <span style="font-size:.75rem;color:#6b7280;">{wc} words · {cc} chars</span>
        <span class="badge {q}">{ql}</span></div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Step 2: Voice Input ───────────────────────────────────────────────────
    st.markdown("""<div class="card">
        <div class="card-title"><span class="step-label">2</span> Voice Reference Sample</div>""",
        unsafe_allow_html=True)

    input_tab = st.radio("Input method", ["🎤 Record Now (Recommended)", "📁 Upload File"],
                         horizontal=True, label_visibility="collapsed")

    speaker_bytes = None
    speaker_ext   = "wav"

    # ── TAB A: built-in mic recorder ─────────────────────────────────────────
    if input_tab == "🎤 Record Now (Recommended)":
        st.markdown("""<div class="rec-box">
            <div style="font-size:1.5rem;">🎙️</div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;color:#a78bfa;margin:6px 0 4px;">
                Record directly in your browser
            </div></div>""", unsafe_allow_html=True)

        audio_val = st.audio_input(
            label="Click the mic to record your voice",
            label_visibility="visible",
            key="mic_input"
        )

        st.markdown("""<div class="rec-tip">
            💡 <b>Tips for best cloning accuracy:</b><br>
            • Speak clearly for <b>5–15 seconds</b><br>
            • Use a quiet room — no background noise<br>
            • Natural speaking pace works best<br>
            • Read a paragraph of text out loud
        </div>""", unsafe_allow_html=True)

        if audio_val is not None:
            speaker_bytes = audio_val.read()
            speaker_ext   = "wav"
            st.markdown('<span class="badge badge-success">✅ Recording captured — ready to clone!</span>',
                        unsafe_allow_html=True)

    # ── TAB B: file upload ────────────────────────────────────────────────────
    else:
        st.markdown("""<div class="rec-tip" style="margin-bottom:10px;">
            ⚠️ <b>File must be a valid audio file.</b> If you get errors, use the <b>Record Now</b> tab instead —
            it always produces a clean WAV that works perfectly.
        </div>""", unsafe_allow_html=True)

        multi_mode = st.checkbox("Upload multiple samples (better accuracy)", value=False)
        uploaded   = st.file_uploader(
            "Drop .wav / .mp3 / .ogg file(s)",
            type=["wav","mp3","ogg","flac","m4a"],
            accept_multiple_files=multi_mode
        )

        if uploaded:
            files = uploaded if isinstance(uploaded, list) else [uploaded]
            # Save all uploads, merge if needed, read back as bytes
            saved = []
            for i, uf in enumerate(files):
                p = os.path.join(TEMP_DIR, f"upload_{i}_{uf.name}")
                raw = uf.read()
                with open(p, "wb") as f:
                    f.write(raw)
                saved.append(p)

            if len(saved) == 1:
                with open(saved[0], "rb") as f:
                    speaker_bytes = f.read()
                speaker_ext = saved[0].rsplit(".", 1)[-1]
            else:
                try:
                    merged_path = merge_audio_files(saved, os.path.join(TEMP_DIR, "pre_merge.wav"))
                    with open(merged_path, "rb") as f:
                        speaker_bytes = f.read()
                    speaker_ext = "wav"
                    st.markdown(f'<span class="badge badge-success">✓ {len(saved)} files merged</span>',
                                unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ Merge failed: {e}")

            if speaker_bytes:
                q_badge = "badge-success" if len(files)>=3 else "badge-warn"
                st.markdown(f'<span class="badge {q_badge}">✓ {len(files)} file(s) ready</span>',
                            unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Step 3: Generate ──────────────────────────────────────────────────────
    st.markdown("""<div class="card" style="border-color:rgba(92,110,248,.4);">
        <div class="card-title"><span class="step-label">3</span> Generate Cloned Speech</div>""",
        unsafe_allow_html=True)

    lang_code = SUPPORTED_LANGUAGES[language]
    st.markdown(f"""<div style="display:flex;gap:10px;margin-bottom:16px;flex-wrap:wrap;">
        <span class="badge badge-info">🌐 {language}</span>
        <span class="badge badge-info">⚡ Speed {speed}x</span>
    </div>""", unsafe_allow_html=True)

    generate_btn = st.button("🎙️  Generate Voice Clone", key="generate")
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT PANEL – analysis of the current speaker sample
# ══════════════════════════════════════════════════════════════════════════════
with col_right:

    if speaker_bytes:
        # Write bytes to a temp file for analysis
        preview_path = os.path.join(TEMP_DIR, f"preview.{speaker_ext}")
        with open(preview_path, "wb") as f:
            f.write(speaker_bytes)

        st.markdown("""<div class="card">
            <div class="card-title">📊 Reference Audio Analysis</div>""", unsafe_allow_html=True)

        try:
            stats = analyze_audio(preview_path)
            st.markdown(f"""<div class="metrics-row">
                <div class="metric-chip"><div class="val">{stats['duration_sec']}s</div><div class="lbl">Duration</div></div>
                <div class="metric-chip"><div class="val">{stats['sample_rate']//1000}kHz</div><div class="lbl">Sample Rate</div></div>
                <div class="metric-chip"><div class="val">{stats['snr_db']}</div><div class="lbl">SNR (dB)</div></div>
                <div class="metric-chip"><div class="val">{int(stats['spectral_centroid_hz']//100)/10}k</div><div class="lbl">Spectral</div></div>
            </div>""", unsafe_allow_html=True)

            q_score = min(100, int(
                (min(stats['duration_sec'],10)/10)*40 +
                (min(max(stats['snr_db'],0),20)/20)*40 +
                (1-min(stats['zero_crossing_rate']*10,1))*20))
            qc = "badge-success" if q_score>=70 else "badge-warn" if q_score>=40 else "badge-danger"
            qt = "High Quality" if q_score>=70 else "Moderate Quality" if q_score>=40 else "Low Quality"
            st.markdown(f'<div style="margin-top:12px;"><span class="badge {qc}">Quality Score: {q_score}/100 · {qt}</span></div>',
                        unsafe_allow_html=True)

            wv  = get_waveform_data(preview_path)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wv["times"], y=wv["samples"], mode="lines",
                line=dict(color="#5c6ef8",width=1), fill="tozeroy",
                fillcolor="rgba(92,110,248,0.08)"))
            fig.update_layout(height=130, margin=dict(l=0,r=0,t=8,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        except Exception as e:
            st.warning(f"Preview analysis failed: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""<div class="card" style="text-align:center;padding:48px 24px;">
            <div style="font-size:3rem;margin-bottom:16px;">🎤</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:#9ca3af;font-weight:600;">
                Record or upload a voice sample
            </div>
            <div style="font-size:.85rem;color:#6b7280;margin-top:8px;">
                Waveform & quality analysis will appear here
            </div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  GENERATION LOGIC
# ══════════════════════════════════════════════════════════════════════════════
if generate_btn:
    errors = []
    if not text_input.strip():        errors.append("Please enter some text.")
    if speaker_bytes is None:         errors.append("Please record or upload a voice sample.")

    if errors:
        for e in errors: st.error(f"❌ {e}")
    else:
        # Save speaker bytes → clean WAV
        raw_path = os.path.join(TEMP_DIR, f"raw_speaker.{speaker_ext}")
        with open(raw_path, "wb") as f:
            f.write(speaker_bytes)

        progress = st.progress(0, text="Initializing…")
        status   = st.empty()

        with st.spinner(""):
            progress.progress(15, text="🔧 Preprocessing audio…")
            status.markdown('<span class="badge badge-info">● Cleaning & normalizing audio</span>',
                            unsafe_allow_html=True)
            try:
                speaker_wav = clean_audio(raw_path, os.path.join(TEMP_DIR, "speaker_clean.wav"))
            except ValueError as e:
                st.error(f"❌ Audio Error: {e}")
                st.stop()

            progress.progress(40, text="🧠 Loading XTTS v2 model…")
            status.markdown('<span class="badge badge-info">● Loading model (first run ~30s)</span>',
                            unsafe_allow_html=True)

            out_path = os.path.join(OUTPUT_DIR, f"clone_{int(time.time())}.wav")

            progress.progress(65, text="🎙️ Synthesizing voice…")
            status.markdown('<span class="badge badge-warn">● Generating cloned speech…</span>',
                            unsafe_allow_html=True)

            result = generate_speech(
                text=text_input, speaker_wav=speaker_wav,
                language_code=lang_code, output_path=out_path, speed=speed)

            progress.progress(100, text="✅ Done!")
            status.empty()

        if result["success"]:
            st.session_state.generation_count += 1
            st.session_state.history.append({
                "text": text_input, "lang": language,
                "gen_time": result["generation_time_sec"],
                "dur": result["duration_sec"], "path": result["output_path"]})

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"""<div class="card" style="border-color:rgba(34,211,165,.4);">
                <div class="card-title">🔊 Generated Voice Output</div>
                <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;">
                    <span class="badge badge-success">✅ Complete</span>
                    <span class="badge badge-info">⏱ {result['generation_time_sec']}s</span>
                    <span class="badge badge-info">🔊 {result['duration_sec']}s audio</span>
                    <span class="badge badge-info">🌐 {language}</span>
                </div></div>""", unsafe_allow_html=True)

            with open(result["output_path"], "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/wav")
            st.download_button("⬇️  Download Generated Audio", data=audio_bytes,
                file_name=f"voiceforge_{int(time.time())}.wav", mime="audio/wav")

            try:
                wv2  = get_waveform_data(result["output_path"])
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=wv2["times"], y=wv2["samples"], mode="lines",
                    line=dict(color="#22d3a5",width=1), fill="tozeroy",
                    fillcolor="rgba(34,211,165,0.08)"))
                fig2.update_layout(height=110, margin=dict(l=0,r=0,t=4,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                    showlegend=False)
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
            except Exception:
                pass

        else:
            st.error(f"❌ Generation failed: {result['error']}")
            if "cuda" in str(result["error"]).lower():
                st.info("💡 GPU unavailable — TTS will run on CPU (slower but works).")