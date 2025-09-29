import requests
import json

BASE_URL = "http://localhost:5000"

def call_chat(query: str):
    payload = {
        "query": query,
        "use_agent": True
    }
    r = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=60)
    print(f"Status: {r.status_code}")
    try:
        data = r.json()
    except Exception:
        print(r.text)
        return
    print(json.dumps({
        "query": query,
        "answer": data.get("answer"),
        "source": data.get("source"),
        "stats": data.get("stats"),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    print("测试1：文档建立日期")
    call_chat("文档建立日期是什么？")

    print("\n测试2：文件作者")
    call_chat("文件作者是谁？")