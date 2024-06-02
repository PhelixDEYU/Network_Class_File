from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import openpyxl
import os
import datetime

app = Flask(__name__)

# 加載自動語音識別模型
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")

# Summarizer pipeline
summarizer_pipeline = pipeline("summarization")

# 設定 Excel 檔案路徑
excel_file = "user_actions.xlsx"

# 如果 Excel 檔案不存在，則創建一個新的
if not os.path.exists(excel_file):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Timestamp", "Action", "Details"])
    wb.save(excel_file)

@app.route('/')
def index():
    return render_template('tf.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    audio_file = request.files['audio_file']
    audio_path = "uploaded_audio.wav"
    audio_file.save(audio_path)

    # 轉錄音檔
    result = asr_pipeline(audio_path)  # 確保這裡傳遞了 language='zh' 參數
    text = result["text"]

    # 保存轉錄文本
    with open("original_lyrics.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    # 紀錄動作到 Excel
    record_action("Upload Audio", "Uploaded audio file")

    return jsonify({"transcription": text})

@app.route('/summarize', methods=['POST'])
def summarize():
    with open("original_lyrics.txt", "r", encoding="utf-8") as file:
        text = file.read()

    # 生成摘要
    summary = summarizer_pipeline(text, max_length=100, min_length=50, do_sample=False)

    # 紀錄動作到 Excel
    record_action("Generate Summary", "Generated summary")

    return jsonify({"summary": summary[0]['summary_text']})

def record_action(action, details):
    # 寫入動作記錄到 Excel
    wb = openpyxl.load_workbook(excel_file)
    ws = wb.active
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ws.append([timestamp, action, details])
    wb.save(excel_file)

if __name__ == '__main__':
    app.run(debug=True)
