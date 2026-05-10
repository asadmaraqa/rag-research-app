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
    mode  = data.get("mode", "traditional")  # "traditional", "single", or "multi"
    if not query:
        return jsonify({"error": "Empty query."})

    # ── Input guardrail ───────────────────────────────────────────────────────
    is_safe, reason = validate_input(query)
    if not is_safe:
        print(f"[GUARDRAIL] Input blocked: {reason}")
        return jsonify({"error": f"Query blocked by safety guardrail: {reason}", "sources": []}), 400

    try:
        # Route to the chosen pipeline based on the mode sent from the UI
        if mode == "multi":
            result = multi_agent_ask(query)
        elif mode == "single":
            result = single_agent_ask(query)
        elif mode == "react":
            result = react_ask(query)
        else:
            result = traditional_ask(query)

        # ── Output guardrail ──────────────────────────────────────────────────
        answer  = result.get("answer", "")
        context = " ".join(
            s.get("page_content", "") for s in result.get("sources", []) if isinstance(s, dict)
        )
        is_ok, issue = validate_output(query, answer, context)
        if not is_ok:
            print(f"[GUARDRAIL] Output flagged: {issue}")
            result["guardrail_warning"] = issue

        return jsonify({**result, "mode": mode})

    except Exception as e:
        # Catch any pipeline error and return it as a readable message
        print(f"[ERROR] Pipeline failed: {e}")
        return jsonify({"error": f"Pipeline error: {str(e)}", "sources": []}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, port=port, threaded=True)
