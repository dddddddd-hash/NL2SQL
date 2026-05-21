from sentence_transformers import SentenceTransformer

print("开始下载 bge-m3 模型，首次约需几分钟...")
model = SentenceTransformer("BAAI/bge-m3", cache_folder="./models/bge-m3")
print("下载完成")
