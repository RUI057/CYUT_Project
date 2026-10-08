import platform
import os
from PIL import ImageFont, Image, ImageDraw
import cv2
import numpy as np

def get_font(size: int) -> ImageFont.FreeTypeFont:
    """
    自動依作業系統選擇中文字型，Windows / macOS 都適用。
    找不到任何字型時 fallback 到預設字型。
    """
    system = platform.system()

    candidates = []

    if system == "Windows":
        candidates = [
            "C:/Windows/Fonts/msjh.ttc",       # 微軟正黑體
            "C:/Windows/Fonts/mingliu.ttc",     # 新細明體
            "C:/Windows/Fonts/kaiu.ttf",        # 標楷體
        ]
    elif system == "Darwin":  # macOS
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",              # 蘋方（最優先）
            "/System/Library/Fonts/STHeiti Light.ttc",         # 黑體-簡
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Microsoft/Microsoft JhengHei.ttf", # 有裝 Office 才有
        ]
    else:  # Linux
        candidates = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]

    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    # 都找不到就用預設（不支援中文但不會 crash）
    print("[警告] 找不到中文字型，畫面中文可能顯示亂碼")
    return ImageFont.load_default()


def put_text(frame: np.ndarray, text: str, pos: tuple,
             font: ImageFont.FreeTypeFont,
             color: tuple = (255, 255, 255)) -> np.ndarray:
    """在 OpenCV BGR frame 上疊加中文文字，回傳 BGR frame"""
    return put_texts(frame, [(text, pos, font, color)])


def put_texts(frame: np.ndarray, items) -> np.ndarray:
    """一次疊加多段中文文字：items = [(text, pos, font, color[, outline]), ...]，回傳 BGR frame。

    color 是 RGB（文字是在轉成 RGB 的圖上畫的）。選填的 outline 是黑色描邊的粗細（像素），
    亮色字疊在淺色背景上會看不清楚，描邊讓任何背景都讀得到。

    每一幀有好幾段文字時一定要用這個：put_text 每呼叫一次，整張畫面就要
    BGR→RGB→PIL→RGB→BGR 轉換一次。實測 1080p 畫 8 段要 40 ms，占收集迴圈 77% 的時間，
    幀率掉到 19 fps，快速手勢時 MediaPipe 就容易跟丟。合併成一次轉換只要幾 ms。
    """
    if not items:
        return frame
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    for text, pos, font, color, *outline in items:
        draw.text(pos, text, font=font, fill=color,
                  stroke_width=outline[0] if outline else 0, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
