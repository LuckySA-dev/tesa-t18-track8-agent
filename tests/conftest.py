import sys
from pathlib import Path

T32 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(T32.parent / "vendor" / "TesaMarket-build"))
sys.path.insert(0, str(T32))          # t3-2 modules win over agent/ ones
