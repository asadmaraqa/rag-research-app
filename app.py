import os
import traceback
from dotenv import load_dotenv
load_dotenv()

print("Starting app imports...")
from functools import wraps
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
print("Flask imported")

try:
    from src.data_loader import load_all_documents
    from src.vectorstore import build_vectorstore, vectorstore_exists
    from src.search import ask as traditional_ask
    from src.agent import ask as single_agent_ask
    from src.multi_agent import ask as multi_agent_ask
    from src.react_agent import ask as react_ask
    from src.guardrails import validate_input, validate_output
    print("All src imports successful")
except Exception as e:
    print(f"IMPORT ERROR: {e}")
    traceback.print_exc()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production")

APP_PASSWORD = os.environ.get("APP_PASSWORD", "testItsMe92")


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == APP_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("index"))
        error = "Incorrect password. Try again."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    """Save uploaded files and rebuild the FAISS vector index."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"message": "No files received."}), 400

    saved = []
    for f in files:
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
        saved.append(f.filename)

    # Load all documents from the upload folder and index them
    docs = load_all_documents(UPLOAD_DIR)
    if not docs:
        return jsonify({"message": "Files saved but no content could be extracted."})

    build_vectorstore(docs)
    return jsonify({"message": f"Indexed {len(docs)} chunks from: {', '.join(saved)}"})


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    """Accept a question and return an answer grounded in uploaded documents."""
    if not vectorstore_exists():
        return jsonify({"error": "No documents indexed yet. Please upload files first."})

    data = request.get_json() or {}
    query = data.get("query", "").strip()
    mode  = data.get("mode", "traditional")  # "traditional", "single", "multi", or "react"
    if not query:
        return jsonify({"error": "Empty query."})

    # ── Input guardrail ───────────────────────────────────────────────────────
    is_safe, reason = validate_input(query)
    if not is_safe:
        print(f"[GUARDRAIL] Input blocked: {reason}")
        return jsonify({"error": f"Query blocked by safety guardrail: {reason}", "sources": []}), 400

    # Load conversation history for this session (last 20 turns = 10 exchanges)
    history = session.get("chat_history", [])

    try:
        # Route to the chosen pipeline, passing history each time
        if mode == "multi":
            result = multi_agent_ask(query, chat_history=history)
        elif mode == "single":
            result = single_agent_ask(query, chat_history=history)
        elif mode == "react":
            result = react_ask(query, chat_history=history)
        else:
            result = traditional_ask(query, chat_history=history)

        # ── Output guardrail ──────────────────────────────────────────────────
        answer  = result.get("answer", "")
        context = " ".join(
            s.get("page_content", "") for s in result.get("sources", []) if isinstance(s, dict)
        )
        is_ok, issue = validate_output(query, answer, context)
        if not is_ok:
            print(f"[GUARDRAIL] Output flagged: {issue}")
            result["guardrail_warning"] = issue

        # Persist this exchange to the session (keep last 20 messages = 10 turns)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        session["chat_history"] = history[-20:]
        session.modified = True

        return jsonify({**result, "mode": mode})

    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        return jsonify({"error": f"Pipeline error: {str(e)}", "sources": []}), 500


@app.route("/eval", methods=["POST"])
@login_required
def run_eval():
    """Trigger a LangSmith evaluation run and return aggregated scores."""
    data = request.get_json() or {}
    mode = data.get("mode", "traditional")
    if mode not in ("traditional", "single", "multi", "react"):
        return jsonify({"error": f"Unknown mode '{mode}'."}), 400

    try:
        from eval.run_evals import run_evaluation
        summary = run_evaluation(mode)
        return jsonify({"mode": mode, "results": summary})
    except Exception as e:
        print(f"[ERROR] Eval failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
@login_required
def clear_history():
    """Clear the conversation history for the current session."""
    session.pop("chat_history", None)
    session.modified = True
    return jsonify({"message": "Conversation history cleared."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, port=port, threaded=True)
