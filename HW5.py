import subprocess
subprocess.run(["pip", "install", "-q", "-U", "InstructorEmbedding"])
subprocess.run(["pip", "install", "huggingface_hub", "-q"])
subprocess.run(["pip", "install", "langchain==0.1.2"])
subprocess.run(["pip", "install", "sentence_transformers==2.2.2"])
subprocess.run(["pip", "install", "transformers"])
subprocess.run(["pip", "install", "datasets"])
subprocess.run(["pip", "install", "--upgrade", "pip"])
subprocess.run(["pip", "install", "langchain_community"])

import subprocess
import os
import datetime
import pandas as pd

# 定義一個函式來紀錄時間戳記到 Excel 中

def log_timestamp_to_excel(timestamp, operation):
    # 讀取已存在的 Excel 檔案或創建一個新的
    try:
        df = pd.read_excel("Appilication Record.xlsx")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Time", "Action"])

    # 新增一行到 DataFrame 中
    new_row = pd.DataFrame({"Time": [timestamp], "Action": [operation]})
    df = pd.concat([df, new_row], ignore_index=True)

    # 將 DataFrame 寫入到 Excel 檔案中
    df.to_excel("Appilication Record.xlsx", index=False)

# 紀錄程式開始執行的時間
start_time = datetime.datetime.now()

# 紀錄安裝套件的時間
log_timestamp_to_excel(datetime.datetime.now(), "安裝相關套件")

# 設定 Hugging Face Hub 的 API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# 載入模型和處理器
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
import torch
from datasets import load_dataset

# 紀錄載入模型和處理器的時間
log_timestamp_to_excel(datetime.datetime.now(), "載入模型和處理器")

# 創建自動語音識別 pipeline
pipe = pipeline("automatic-speech-recognition", model="washeed/audio-transcribe")

# 載入模型和處理器
processor = AutoProcessor.from_pretrained("washeed/audio-transcribe")
model = AutoModelForSpeechSeq2Seq.from_pretrained("washeed/audio-transcribe")

# 載入音訊文件
audio_file = "/Users/deyu/Documents/GitHub/Network_Class_File/Network_Class_File/illusion.wav"
waveform, sample_rate = librosa.load(audio_file, sr=None)

# 提取 MFCC 特徵
mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate)

# 打印特徵矩陣的形狀
print("MFCCs shape:", mfccs.shape)

# 配置模型和 pipeline 的設定
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "washeed/audio-transcribe"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

# 加載數據集
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]

# 使用 pipeline 進行語音識別
result = pipe(sample)
print("識別結果:", result["text"])

# 紀錄自動語音識別的時間
log_timestamp_to_excel(datetime.datetime.now(), "自動語音識別")

# 將音訊文件進行識別並保存到文字文件中
result = pipe("illusion.wav", return_timestamps=True)
chunks = result["chunks"]
text = "\n".join(chunk["text"] for chunk in chunks)
output_file = "original_lyrics.txt"
with open(output_file, "w") as f:
    f.write(text)
print(f"已將輸出的文本保存到 {output_file} 文件中")

# 紀錄識別結果保存到文字文件的時間
log_timestamp_to_excel(datetime.datetime.now(), "保存分析結果到txt")

# 紀錄程式結束執行的時間
end_time = datetime.datetime.now()
log_timestamp_to_excel(end_time, "音檔分析程式執行結束")

# 以下是新增的程式碼，用於紀錄
from langchain.vectorstores import FAISS
class CFG:
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    temperature = 0.5
    top_p = 0.95
    repetition_penalty = 1.15
    do_sample = True
    max_new_tokens = 400
    num_return_sequences = 1

    split_chunk_size = 800
    split_overlap = 0
    
    embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'

    k = 3
    
    text_path = '/Users/deyu/Documents/GitHub/Network_Class_File/Network_Class_File/original_lyrics.txt'
    Embeddings_path = './faiss_index_py'

from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id = CFG.model_name,
    model_kwargs={
        "max_new_tokens": CFG.max_new_tokens,
        "temperature": CFG.temperature,
        "top_p": CFG.top_p,
        "repetition_penalty": CFG.repetition_penalty,
        "do_sample": CFG.do_sample,
        "num_return_sequences": CFG.num_return_sequences
    }
) 

from langchain.document_loaders import TextLoader

# 打開文本檔案並讀取內容
with open(CFG.text_path, 'r', encoding='utf-8') as file:
    text = file.read()

# 現在你可以對讀取的文本內容進行後續處理
print("音檔內容:", text)

# 紀錄文本處理的時間
with pd.ExcelWriter("Appilication Record.xlsx", mode='a', engine='openpyxl') as writer:
    pd.DataFrame(data={"音檔內容": [text]}).to_excel(writer, index=False, sheet_name="音檔分析內容")


from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
subprocess.run(["pip", "install", "textsum"])
#hugging face:https://reurl.cc/xLnXee
from textsum.summarize import Summarizer

model_name = "pszemraj/led-large-book-summary"
summarizer = Summarizer(
    model_name_or_path=model_name,  # you can use any Seq2Seq model on the Hub
    token_batch_length=4096,  # tokens to batch summarize at a time, up to 16384
)

from textsum.summarize import Summarizer

# 定義文件路徑
file_path = "/Users/deyu/Documents/GitHub/Network_Class_File/Network_Class_File/original_lyrics.txt"

# 讀取文本內容
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# 創建 Summarizer 實例
summarizer = Summarizer()

# 將文本拆分成句子
sentences = text.split(".")

# 提取前幾個句子作為摘要
summary_sentences = sentences[:3]

# 將摘要句子組合成摘要文本
summary = ". ".join(summary_sentences)

# 生成摘要
result_text = summarizer(summary)
print(f"summary: {result_text}")

# 紀錄生成摘要的時間
with pd.ExcelWriter("Appilication Record.xlsx", mode='a', engine='openpyxl') as writer:
    pd.DataFrame(data={"summary": [result_text]}).to_excel(writer, index=False, sheet_name="Summary")