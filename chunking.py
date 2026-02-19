from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text(file):
    data = file.getvalue()
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        pages = [(p.extract_text() or "") for p in reader.pages]
        return "\n".join(pages)
    return data.decode("utf-8", errors="ignore")

def make_chunks(files):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = []
    metas = []
    ids = []
    i = 0

    for f in files:
        text = extract_text(f)
        parts = splitter.split_text(text)
        for p in parts:
            ids.append(f"c{i}")
            chunks.append(p)
            metas.append({"file": f.name})
            i += 1
    return chunks, metas, ids
