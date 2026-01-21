import os
import sqlite3
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_NAME = "database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("INSERT INTO chats (role, content) VALUES (?, ?)",
                   ("user", user_message))
    conn.commit()

    cursor.execute("SELECT role, content FROM chats")
    history = cursor.fetchall()

    messages = [{"role": r, "content": c} for r, c in history]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    reply = response.choices[0].message.content

    cursor.execute("INSERT INTO chats (role, content) VALUES (?, ?)",
                   ("assistant", reply))
    conn.commit()
    conn.close()

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)