import os
import io
import json
import requests
import gradio as gr
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBED_MODEL_NAME)

analytics = {
    "total_queries": 0,
    "queries_by_doc": {},
    "avg_confidence": [],
}

def extract_text_from_pdf(file_obj):
    reader = PdfReader(file_obj)
    pages_text = []
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except:
            text = ""
        pages_text.append((page_num + 1, text))
    return pages_text

def smart_chunker(pages_text, max_chars=800, overlap=100):
    chunks = []
    meta = []
    for page_num, text in pages_text:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        current = ""
        for p in parts:
            if len(current) + len(p) + 1 <= max_chars:
                current = current + "\n" + p if current else p
            else:
                if current:
                    chunks.append(current)
                    meta.append({"page": page_num})
                    words = current.split()
                    if len(words) > 20:
                        current = " ".join(words[-20:]) + "\n" + p
                    else:
                        current = p
                else:
                    current = p
        if current:
            chunks.append(current)
            meta.append({"page": page_num})
    return chunks, meta

def build_corpus(files):
    docs = []
    all_chunks = []
    all_meta = []
    
    for f in files:
        if f is None:
            continue
        pages_text = extract_text_from_pdf(f)
        chunks, meta = smart_chunker(pages_text)
        
        first_page = pages_text[0][1][:500] if pages_text else ""
        
        docs.append({
            "name": os.path.basename(f.name),
            "pages": len(pages_text),
            "chunks": chunks,
            "chunk_meta": meta,
            "summary": first_page,
            "upload_time": datetime.now().strftime("%H:%M:%S")
        })
        doc_idx = len(docs) - 1
        for i, m in enumerate(meta):
            all_chunks.append(chunks[i])
            all_meta.append({"doc_idx": doc_idx, "page": m["page"]})
    
    if all_chunks:
        embeddings = embedder.encode(all_chunks, convert_to_numpy=True, normalize_embeddings=True)
    else:
        embeddings = np.zeros((0, 384), dtype=np.float32)
    
    return {
        "docs": docs,
        "all_chunks": all_chunks,
        "all_meta": all_meta,
        "embeddings": embeddings,
    }

def retrieve_relevant_chunks(state, query, k=5, product_filter=None, min_score=0.2):
    if state is None or len(state.get("all_chunks", [])) == 0:
        return []
    
    query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    embs = state["embeddings"]
    scores = np.dot(embs, query_emb)
    
    indices = np.argsort(-scores)
    results = []
    for idx in indices:
        score = float(scores[idx])
        if score < min_score:
            continue
        meta = state["all_meta"][idx]
        doc_idx = meta["doc_idx"]
        doc_name = state["docs"][doc_idx]["name"]
        if product_filter and product_filter != "All products":
            if product_filter not in doc_name:
                continue
        chunk_text = state["all_chunks"][idx]
        results.append({
            "chunk": chunk_text,
            "score": score,
            "doc_name": doc_name,
            "page": meta["page"],
        })
        if len(results) >= k:
            break
    return results

def call_groq_llm(context_chunks, question, tone, length, strict_halal, language="English"):
    if not GROQ_API_KEY:
        return "⚠️ Error: GROQ_API_KEY not set. Please add it in Space settings."
    
    context_text = "\n\n".join([
        f"[Document: {c['doc_name']}, Page {c['page']}, Relevance: {c['score']:.2f}]\n{c['chunk'][:500]}" 
        for i, c in enumerate(context_chunks[:3])
    ])
    
    tone_map = {
        "Beginner / Easy": "very simple language with examples",
        "Intermediate": "clear explanation with balanced technical terms",
        "Professional / Scholar": "scholarly tone with precise terminology"
    }
    
    length_map = {
        "Short": "Answer in 2-3 concise sentences only.",
        "Medium": "Answer in 1-2 well-structured paragraphs.",
        "Detailed": "Provide a comprehensive, detailed explanation."
    }
    
    if strict_halal:
        halal_mode = "STRICT MODE: ONLY answer from provided documents. Do not add external knowledge or opinions. If the answer isn't in the documents, say so clearly."
    else:
        halal_mode = "FLEXIBLE MODE: Answer primarily from documents, but you may use general Islamic finance knowledge to provide context and clarity."
    
    system_prompt = f"""You are HalalFinance.AI, an Islamic banking assistant.

Style: {tone_map.get(tone, tone_map['Intermediate'])}
Length: {length_map.get(length, length_map['Medium'])}
Mode: {halal_mode}
Language: {language}

Always cite documents and page numbers."""
    
    user_prompt = f"""Documents:

{context_text}

Question: {question}

Answer:"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 500,
    }
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    
    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        
        if resp.status_code == 400:
            try:
                error_detail = resp.json().get("error", {}).get("message", "Bad request")
                return f"⚠️ API Error 400: {error_detail}"
            except:
                return "⚠️ API Error 400: Check API key."
        elif resp.status_code == 401:
            return "⚠️ Invalid API key."
        elif resp.status_code == 429:
            return "⚠️ Rate limit exceeded."
        elif resp.status_code != 200:
            return f"⚠️ Error {resp.status_code}"
        
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def analyze_pdfs(files):
    if not files:
        return None, "⚠️ Upload at least one PDF", generate_doc_preview([]), ["(no suggestions)"], ["All products"], ""
    
    state = build_corpus(files)
    
    suggestions = [
        "How does this Murabaha product avoid interest?",
        "What are the charges and fees mentioned?",
        "Is there any interest involved?",
        "What happens if payment is late?",
        "What Shariah principles are used?",
        "Explain the profit-sharing mechanism",
        "What are the eligibility criteria?",
        "Compare with conventional banking"
    ]
    
    names = sorted([d["name"] for d in state["docs"]])
    filter_choices = ["All products"] + names
    
    preview = generate_doc_preview(state["docs"])
    
    return state, f"✅ **Processed {len(state['docs'])} documents** with {len(state['all_chunks'])} chunks", preview, suggestions, filter_choices, generate_analytics()

def generate_doc_preview(docs):
    if not docs:
        return "### 📂 Documents\n\nNo documents uploaded yet."
    
    html = "### 📂 Document Library\n\n"
    for d in docs:
        summary = d.get("summary", "")[:200] + "..." if d.get("summary") else "No preview"
        html += f"""
**📄 {d['name']}**
- Pages: {d['pages']}
- Chunks: {len(d['chunks'])}
- Uploaded: {d.get('upload_time', 'N/A')}

*Preview:* {summary}

---
"""
    return html

def format_sources(context_chunks):
    if not context_chunks:
        return "No sources"
    
    lines = []
    for i, c in enumerate(context_chunks):
        conf = c["score"]
        if conf >= 0.7:
            badge = "🟢 Very High"
        elif conf >= 0.5:
            badge = "🟡 High"
        elif conf >= 0.3:
            badge = "🟠 Medium"
        else:
            badge = "🔴 Low"
        lines.append(f"{badge} • {c['doc_name']} (Page {c['page']})")
    return "\n".join(lines)

def format_chat_history(history):
    if not history:
        return "<div style='padding:3rem; text-align:center; color:#888;'>💬 No messages yet. Upload PDFs and ask!</div>"
    
    html = "<div style='display:flex; flex-direction:column; gap:1.25rem;'>"
    for idx, (user_msg, bot_msg) in enumerate(history):
        timestamp = datetime.now().strftime("%H:%M")
        user_safe = user_msg.replace("<", "&lt;").replace(">", "&gt;")
        bot_safe = bot_msg.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        
        html += f"""
        <div style='display:flex; flex-direction:column; gap:0.75rem;'>
            <div style='align-self:flex-end; background:linear-gradient(135deg, #059669, #047857); color:white; padding:1rem 1.5rem; border-radius:20px 20px 6px 20px; max-width:70%; box-shadow:0 4px 16px rgba(5,150,105,0.35);'>
                <div style='font-size:0.75rem; opacity:0.8; margin-bottom:0.25rem;'>{timestamp}</div>
                <strong>You:</strong><br><div style='margin-top:0.5rem;'>{user_safe}</div>
            </div>
            <div style='align-self:flex-start; background:rgba(255,255,255,0.06); backdrop-filter:blur(12px); color:white; padding:1rem 1.5rem; border-radius:20px 20px 20px 6px; max-width:85%; border:1px solid rgba(255,255,255,0.12);'>
                <strong style='color:#10b981;'>🕋 HalalFinance.AI</strong>
                <div style='margin-top:0.75rem; line-height:1.6;'>{bot_safe}</div>
            </div>
        </div>
        """
    html += "</div>"
    return html

def answer_question(question, tone, length, strict_halal, product_filter, language, state, chat_history):
    if not question.strip():
        return "", chat_history, state, "⚠️ Enter question", format_chat_history(chat_history), generate_analytics()
    
    if state is None or len(state.get("all_chunks", [])) == 0:
        return "", chat_history, state, "⚠️ Upload PDFs first", format_chat_history(chat_history), generate_analytics()
    
    analytics["total_queries"] += 1
    
    ctx_chunks = retrieve_relevant_chunks(state, question, k=5, product_filter=product_filter)
    if not ctx_chunks:
        bot_msg = "❌ No relevant information found."
        new_history = chat_history + [[question, bot_msg]]
        return "", new_history, state, "⚠️ No context", format_chat_history(new_history), generate_analytics()
    
    for chunk in ctx_chunks:
        doc_name = chunk["doc_name"]
        analytics["queries_by_doc"][doc_name] = analytics["queries_by_doc"].get(doc_name, 0) + 1
    
    avg_conf = sum(c["score"] for c in ctx_chunks) / len(ctx_chunks)
    analytics["avg_confidence"].append(avg_conf)
    
    answer = call_groq_llm(ctx_chunks, question, tone, length, strict_halal, language)
    sources = format_sources(ctx_chunks)
    full_answer = f"{answer}\n\n---\n\n**📚 Sources:**\n{sources}"
    
    new_history = chat_history + [[question, full_answer]]
    return "", new_history, state, "✅ Generated", format_chat_history(new_history), generate_analytics()

def clear_chat():
    return "", [], "🗑️ Cleared", "<div style='padding:2rem; text-align:center; color:#888;'>Chat cleared!</div>", generate_analytics()

def generate_analytics():
    if analytics["total_queries"] == 0:
        return "### 📊 Analytics\n\nNo queries yet."
    
    avg_conf = sum(analytics["avg_confidence"]) / len(analytics["avg_confidence"]) if analytics["avg_confidence"] else 0
    
    report = f"""### 📊 Usage Analytics

**Total Queries:** {analytics['total_queries']}
**Average Confidence:** {avg_conf:.2%}

**Most Queried Documents:**
"""
    
    sorted_docs = sorted(analytics["queries_by_doc"].items(), key=lambda x: x[1], reverse=True)
    for doc, count in sorted_docs[:5]:
        report += f"\n- {doc}: {count} queries"
    
    return report

def download_chat(chat_history):
    if not chat_history or len(chat_history) == 0:
        return None
    
    filename = f"/tmp/HalalFinanceAI_Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("HalalFinance.AI - Chat History Export\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        for i, (u, b) in enumerate(chat_history, start=1):
            f.write(f"\n{'─' * 70}\n")
            f.write(f"Question {i}:\n{u}\n\n")
            f.write(f"Answer {i}:\n{b}\n")
            f.write(f"{'─' * 70}\n")
        
        f.write(f"\n\n{'=' * 70}\n")
        f.write(f"Total Q&A Pairs: {len(chat_history)}\n")
        f.write("=" * 70 + "\n")
    
    return filename

def export_analytics():
    if analytics["total_queries"] == 0:
        return None
    
    filename = f"/tmp/analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    data = {
        "total_queries": analytics["total_queries"],
        "avg_confidence": sum(analytics["avg_confidence"]) / len(analytics["avg_confidence"]) if analytics["avg_confidence"] else 0,
        "queries_by_doc": analytics["queries_by_doc"],
        "export_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    return filename

def export_all_data(state, chat_history):
    """Export complete application data including corpus, chat history, and analytics"""
    if state is None:
        return None
    
    filename = f"/tmp/HalalFinanceAI_FullData_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Prepare document data without embeddings (too large)
    docs_export = []
    for doc in state.get("docs", []):
        docs_export.append({
            "name": doc["name"],
            "pages": doc["pages"],
            "chunks_count": len(doc["chunks"]),
            "upload_time": doc.get("upload_time", "N/A"),
            "summary": doc.get("summary", "")[:500]
        })
    
    # Prepare full export
    full_data = {
        "export_metadata": {
            "export_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "app_version": "1.0",
            "total_documents": len(state.get("docs", [])),
            "total_chunks": len(state.get("all_chunks", []))
        },
        "documents": docs_export,
        "chat_history": [
            {
                "question": pair[0],
                "answer": pair[1],
                "index": i+1
            }
            for i, pair in enumerate(chat_history)
        ],
        "analytics": {
            "total_queries": analytics["total_queries"],
            "avg_confidence": sum(analytics["avg_confidence"]) / len(analytics["avg_confidence"]) if analytics["avg_confidence"] else 0,
            "queries_by_doc": analytics["queries_by_doc"],
            "confidence_history": analytics["avg_confidence"]
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
    
    return filename

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1600px !important; margin: auto; }
body { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1729 100%); background-attachment: fixed; }
.hero-header { background: linear-gradient(135deg, #047857 0%, #059669 50%, #10b981 100%); padding: 3rem 2rem; border-radius: 24px; text-align: center; margin-bottom: 2.5rem; box-shadow: 0 25px 70px rgba(5,150,105,0.45); }
.feature-card { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 16px !important; padding: 1.5rem !important; backdrop-filter: blur(20px) !important; margin-bottom: 1.25rem !important; transition: all 0.3s ease !important; }
.feature-card:hover { background: rgba(255,255,255,0.06) !important; border-color: rgba(16,185,129,0.3) !important; transform: translateY(-2px); }
.chat-display { background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 1.5rem; min-height: 550px; max-height: 550px; overflow-y: auto; }
button { transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important; }
label { color: rgba(255,255,255,0.95) !important; font-weight: 500 !important; }
input, select, textarea { background: rgba(255,255,255,0.07) !important; border: 1px solid rgba(255,255,255,0.15) !important; border-radius: 12px !important; color: white !important; padding: 0.875rem !important; }
"""

with gr.Blocks(css=css, title="HalalFinance.AI", theme=gr.themes.Soft(primary_hue="emerald")) as demo:
    
    gr.HTML("""
    <div class="hero-header">
        <h1 style="margin:0; color:white; font-size:3.25rem; font-weight:800;">🕋 HalalFinance.AI</h1>
        <p style="margin:1rem 0 0 0; color:rgba(255,255,255,0.95); font-size:1.3rem; font-weight:600;">Advanced Islamic Banking Document Intelligence</p>
        <p style="margin:0.5rem 0 0 0; color:rgba(255,255,255,0.85);">RAG • Sentence Transformers • Groq LLM</p>
    </div>
    """)
    
    rag_state = gr.State(value=None)
    chat_state = gr.State(value=[])
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="feature-card"):
                gr.Markdown("### 📤 Upload")
                pdf_upload = gr.File(label="", file_types=[".pdf"], file_count="multiple")
                process_btn = gr.Button("🔄 Process", variant="primary", size="lg")
                status = gr.Markdown("📌 Ready")
            
            with gr.Group(elem_classes="feature-card"):
                gr.Markdown("### ⚙️ Controls")
                product_filter = gr.Dropdown(label="🏦 Filter", choices=["All products"], value="All products")
                tone = gr.Dropdown(label="🎯 Level", choices=["Beginner / Easy", "Intermediate", "Professional / Scholar"], value="Beginner / Easy")
                length = gr.Radio(label="📏 Length", choices=["Short", "Medium", "Detailed"], value="Medium")
                language = gr.Dropdown(label="🌐 Language", choices=["English", "English (Urdu-friendly)", "Urdu"], value="English")
                strict_halal = gr.Checkbox(
                    label="✅ Strict Document Mode", 
                    value=True,
                    info="Toggle: ON = Only use documents | OFF = Allow general knowledge",
                    interactive=True
                )
            
            with gr.Group(elem_classes="feature-card"):
                gr.Markdown("### 🛠️ Export Options")
                clear_btn = gr.Button("🗑️ Clear Chat", size="sm", variant="secondary")
                download_btn = gr.Button("💾 Download Chat History", size="sm")
                analytics_btn = gr.Button("📊 Download Analytics", size="sm")
                export_all_btn = gr.Button("📦 Export All Data (Complete)", size="sm", variant="primary")
        
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("💬 Chat"):
                    chat_display = gr.HTML("<div style='padding:3rem; text-align:center; color:#888;'>💬 Upload PDFs to start!</div>", elem_classes="chat-display")
                    
                    gr.Markdown("### 💡 Suggestions")
                    suggested = gr.Radio(choices=["(no suggestions)"], label="", value="(no suggestions)")
                    
                    with gr.Row():
                        user_input = gr.Textbox(label="", placeholder="💬 Ask about documents...", scale=5, show_label=False)
                        ask_btn = gr.Button("Send ➤", variant="primary", scale=1)
                
                with gr.Tab("📂 Documents"):
                    doc_gallery = gr.Markdown("### 📂 Library\n\nNo documents yet.")
                
                with gr.Tab("📊 Analytics"):
                    analytics_display = gr.Markdown("### 📊 Analytics\n\nNo queries yet.")
    
    gr.Markdown("""
    <div style="text-align:center; margin-top:2.5rem; padding:1.5rem; background:rgba(255,255,255,0.03); border-radius:12px; color:rgba(255,255,255,0.6); font-size:0.9rem;">
        <strong style="color:rgba(255,255,255,0.8);">✅ Features:</strong> Multi-PDF • Semantic Chunking • Sentence-Transformers • Vector Search • Source References • Chat History • PDF Preview • Analytics & Logging • Complete Data Export • Toggleable Strict Mode
    </div>
    """)
    
    # Event handlers
    def on_process(files):
        result = analyze_pdfs(files)
        if result[0] is None:
            return None, result[1], result[2], gr.update(choices=result[3], value=result[3][0]), gr.update(choices=result[4], value=result[4][0]), result[5]
        return result[0], result[1], result[2], gr.update(choices=result[3], value=result[3][0]), gr.update(choices=result[4], value=result[4][0]), result[5]
    
    process_btn.click(on_process, [pdf_upload], [rag_state, status, doc_gallery, suggested, product_filter, analytics_display])
    
    def on_suggest(choice, t, l, s, p, lang, st, ch):
        if choice == "(no suggestions)":
            return "", ch, st, "⚠️ Select valid", format_chat_history(ch), generate_analytics()
        return answer_question(choice, t, l, s, p, lang, st, ch)
    
    suggested.change(on_suggest, [suggested, tone, length, strict_halal, product_filter, language, rag_state, chat_state], [user_input, chat_state, rag_state, status, chat_display, analytics_display])
    
    ask_btn.click(answer_question, [user_input, tone, length, strict_halal, product_filter, language, rag_state, chat_state], [user_input, chat_state, rag_state, status, chat_display, analytics_display])
    user_input.submit(answer_question, [user_input, tone, length, strict_halal, product_filter, language, rag_state, chat_state], [user_input, chat_state, rag_state, status, chat_display, analytics_display])
    
    clear_btn.click(clear_chat, None, [user_input, chat_state, status, chat_display, analytics_display])
    
    # Export handlers
    download_btn.click(
        fn=download_chat,
        inputs=[chat_state],
        outputs=gr.File(label="Chat Export")
    )
    
    analytics_btn.click(
        fn=export_analytics,
        inputs=None,
        outputs=gr.File(label="Analytics Export")
    )
    
    export_all_btn.click(
        fn=export_all_data,
        inputs=[rag_state, chat_state],
        outputs=gr.File(label="Complete Data Export")
    )

demo.launch()
