"""本機語言模型（Hugging Face transformers）：不需網路與 API 金鑰的文字生成。

第一次使用會把模型下載到 Hugging Face 快取（~/.cache/huggingface），之後完全離線。
換模型：設定環境變數 LOCAL_LLM=<Hugging Face 模型 id>。

預先下載 / 快速測試：
  python -m src.local_llm            # 下載並載入模型
  python -m src.local_llm 我 明天 學校 去
"""
import os
import sys
import threading
import time

import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_ID = os.getenv("LOCAL_LLM", DEFAULT_MODEL)

_lock   = threading.Lock()
_model  = None
_tok    = None
_device = None
_error  = None
_loading = False


def _pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.device("cuda"), dtype
    return torch.device("cpu"), torch.float32


def load():
    """載入模型（只載一次、執行緒安全）。已下載過就不連網；沒下載過才下載。

    失敗會記住不再重試，避免離線時每句翻譯都卡在下載；重開程式才會再試。
    """
    global _model, _tok, _device, _error, _loading
    with _lock:
        if _model is not None or _error is not None:
            return
        _loading = True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device, dtype = _pick_device()
            try:
                # 先只找本機快取：已下載過就完全離線，不會卡在連線逾時
                tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, dtype=dtype, local_files_only=True)
            except OSError:
                print(f"[本機模型] 第一次使用，下載 {MODEL_ID}（約 3GB，只需一次）...")
                tok = AutoTokenizer.from_pretrained(MODEL_ID)
                model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
            model.to(device).eval()
            _model, _tok, _device, _error = model, tok, device, None
            print(f"[本機模型] {MODEL_ID} 載入完成  裝置：{device}")
        except Exception as e:
            _error = f"{type(e).__name__}: {e}"
            print(f"[本機模型] 載入失敗：{_error}")
        finally:
            _loading = False


def warmup():
    """背景載入模型，讓第一次翻譯不用等載入。"""
    threading.Thread(target=load, daemon=True).start()


def status() -> str:
    if _model is not None:
        return "就緒"
    if _loading:
        return "載入中"
    if _error:
        return "載入失敗"
    return "未載入"


def is_ready() -> bool:
    return _model is not None


def chat(messages, max_new_tokens=48) -> str:
    """messages: [{"role": "system"|"user"|"assistant", "content": str}] → 模型回覆。

    貪婪解碼（結果穩定可重現）。模型沒載入會先載入；載入失敗丟 RuntimeError。
    """
    load()
    if _model is None:
        raise RuntimeError(f"本機模型無法使用：{_error}")
    prompt = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)   # Qwen3 系列關閉思考模式；其他模型忽略
    inputs = _tok(prompt, return_tensors="pt").to(_device)
    with _lock, torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              repetition_penalty=1.05, pad_token_id=_tok.eos_token_id)
    return _tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def chat_candidates(messages, k=6, max_new_tokens=48):
    """束搜尋：一次產生 k 個候選 → [(文字, 分數)]，分數是平均 log 機率（越大越有把握）。

    貪婪解碼只看每一步的第一名，小模型常因此把「我 要 喝」寫成「我要喝水」或混進英文；
    多看幾個候選，再由呼叫端依忠實度挑最好的（見 translate._local）。
    """
    load()
    if _model is None:
        raise RuntimeError(f"本機模型無法使用：{_error}")
    prompt = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
    inputs = _tok(prompt, return_tensors="pt").to(_device)
    with _lock, torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              num_beams=k, num_return_sequences=k, early_stopping=True,
                              repetition_penalty=1.05, pad_token_id=_tok.eos_token_id,
                              return_dict_in_generate=True, output_scores=True)
    n = inputs["input_ids"].shape[1]
    texts = [_tok.decode(seq[n:], skip_special_tokens=True) for seq in out.sequences]
    return list(zip(texts, out.sequences_scores.tolist()))


if __name__ == "__main__":
    # 一律透過 src.local_llm 操作：-m 執行時本檔是 __main__，
    # 直接呼叫本檔的 load() 會和 translate 匯入的 src.local_llm 各載一份模型
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import local_llm, translate
    t = time.time()
    local_llm.load()
    print(f"載入耗時 {time.time() - t:.1f}s")
    if len(sys.argv) > 1:
        t = time.time()
        sentence, backend = translate.polish(sys.argv[1:])
        print(f"{'、'.join(sys.argv[1:])} → {sentence}  （{backend}，{time.time() - t:.2f}s）")
