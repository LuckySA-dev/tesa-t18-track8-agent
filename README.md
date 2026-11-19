# T18 Overclock NeverDie: Track 3 coffee market agent (stage 3, 20 Nov 2026)

TESA Top Gun Rally 2026, Track 3 (sealed-offer coffee market). This folder is the code we submit: the agent that sends
every appraisal and offer by itself, the page we watch it on, and `pack.py` for the hand-in.

## วิธีรัน (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
Copy-Item .env.example .env        # แล้วใส่ MARKET_URL และ TEAM_KEY ของทีม
.venv\Scripts\python.exe monitor.py   # หน้าควบคุม http://127.0.0.1:8770 (ตรวจ, ค่าตั้ง, เริ่ม/หยุด agent, ส่งงาน)
```

หรือรัน agent ตรง ๆ: `.venv\Scripts\python.exe main.py` (ตลาดฝึกเล่นต่อเนื่อง: `--forever`)

ส่งงานเมื่อทีมงานเรียก (ภายใน 10 นาทีหลังปิดตลาด): `.venv\Scripts\python.exe pack.py --send`
`pack.py` zip ทั้งโฟลเดอร์นี้ ไม่รวม `.env` / `.venv` และรวม `logs/` ที่กรรมการใช้เทียบกับข้อมูลที่ตลาดได้รับ

## กติกาที่โค้ดนี้ทำตาม

- agent ส่งผลประเมินและข้อเสนอทุกรอบเอง ไม่มีทางส่ง แก้ หรือยกเลิกข้อเสนอด้วยมือจากหน้า monitor
- รอบแข่งใช้ LLM ได้เฉพาะ gateway ของงาน (model `event-llm`); ค่าตั้งวันนี้ `USE_LLM=0` ใช้กฎอย่างเดียว
- key อยู่ใน `.env` เท่านั้น (git-ignored) หน้า monitor ไม่แสดงค่า key
- เรียกตลาดไม่เกิน 5 ครั้ง/วินาที (รอแล้วลองใหม่เมื่อได้ 429)

## ค่าตั้งของวันแข่ง (`.env`)

| ค่า | ใช้ | ความหมาย |
|---|---|---|
| `STRATEGY` | `adaptive` | k ต่อประเภทผู้ซื้อ เริ่ม 1.12 / 1.00 / 0.90 ปรับ ±0.01 ตามผลรอบก่อน |
| `TACTIC` | `1` | ชั้น tactic: feedback ต่อผู้ซื้อ, โหมด ATTACK/NORMAL/SAFE/RESCUE ตามโควตาต่อรอบ, floor 0.82 |
| `SPEC_P` | `0.5` | ความน่าจะเป็นขั้นต่ำว่าล็อตผ่านสเปคจริงก่อนเสนอ |
| `FREE_TRY` | `1` | โควตาที่เหลือลองล็อตที่เกือบผ่านสเปค |
| `USE_LLM` | `0` | กฎอย่างเดียว |

## ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `main.py` | วงรอบของ agent: อ่าน state, ส่งผลประเมิน, ตัดสินใจและส่งข้อเสนอทุกรอบ, log ทุกการตัดสินใจลง `logs/decisions.jsonl` |
| `strategy.py`, `tactic.py` | การตั้งราคาและเลือกล็อต/ผู้ซื้อ (adaptive + tactic), `guard` ตรวจข้อเสนอก่อนส่ง |
| `planner.py`, `pricing.py`, `pattern.py`, `position.py`, `lpalloc.py` | planner และตัวเรียนเพดานราคา (ใช้เมื่อ `STRATEGY` ว่าง) |
| `appraisal.py`, `market_core.py` | ประเมิน TV จาก sample และกติกาตลาด |
| `market.py`, `memory.py`, `llm.py` | API client, ความจำผลแต่ละรอบ, LLM client (ปิดอยู่) |
| `monitor.py`, `monitor.html`, `control.py`, `fonts/` | หน้าควบคุม: ตรวจความพร้อม, แก้ `.env`, เริ่ม/หยุด agent, ส่งงาน |
| `check.py` | ตรวจการเชื่อมต่อแบบอ่านอย่างเดียว |
| `pack.py` | zip และส่งงาน |
| `tests/` | `python -m pytest -q tests` |
