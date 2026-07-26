"""
LLaDA Web UI — 扩散语言模型文本生成的前端界面（基于 Gradio）。

功能：
- 选择 checkpoints/ 下的模型权重
- 输入 Prompt 或从 TinyStories 数据集随机取样
- 两种重掩码算法：low-confidence（高质量）/ random（基线）
- 两种显示模式：decoding（逐步去掩码）/ correction（观察模型修正预测）
- 流式动画展示扩散解码全过程

运行：python3 ui.py
"""

import glob
import html
import os
import random
import time

import gradio as gr
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from config import CHECKPOINT_DIR, DATA_DIR, MAX_SEQ_LEN, PROMPT_LEN, TOKENIZER_DIR
from helper import get_device, load_model, validate_config

# --------------------------------------------------------------------------
# 缓存：模型 / tokenizer / 数据集只加载一次
# --------------------------------------------------------------------------
_MODELS = {}
_TOKENIZER = None
_DATASET = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    return _TOKENIZER


def get_model(checkpoint_path):
    if checkpoint_path not in _MODELS:
        _MODELS[checkpoint_path] = load_model(checkpoint_path)[0]
    return _MODELS[checkpoint_path]


def get_dataset():
    global _DATASET
    if _DATASET is None:
        _DATASET = load_from_disk(os.path.join(DATA_DIR, "tokenized_tinystories_dataset"))
    return _DATASET


# --------------------------------------------------------------------------
# 流式推理：与 inference.py 相同的两种算法，但逐步 yield 状态而非打印
# --------------------------------------------------------------------------
@torch.no_grad()
def stream_inference(model, prompt_ids, algorithm, sampling_steps, max_length=MAX_SEQ_LEN):
    tokenizer = get_tokenizer()
    device = get_device()
    mask_id = tokenizer.mask_token_id

    prompt_len = len(prompt_ids)
    answer_length = max_length - prompt_len
    response = [mask_id] * answer_length
    prev_correction = None

    def pack(step, new_tokens, num_to_decode, num_changed):
        return {
            "step": step,
            "response": response.copy(),
            "correction_view": prev_correction.copy() if prev_correction else response.copy(),
            "new_tokens": new_tokens,
            "num_masked": sum(1 for r in response if r == mask_id),
            "num_to_decode": num_to_decode,
            "num_changed": num_changed,
        }

    # 初始全掩码状态
    prev_correction = response.copy()
    yield pack(0, set(), 0, 0)

    for step in range(sampling_steps):
        # 线性噪声调度：t 从 1.0 → 0.0，s = t - 1/T
        t = 1.0 - step * (1.0 / sampling_steps)
        s = t - (1.0 / sampling_steps)

        full_input = prompt_ids + response.copy()
        logits = model(torch.tensor([full_input]).to(device))
        pred_ids = torch.argmax(logits, dim=-1)[0][prompt_len:].tolist()

        num_masked = sum(1 for r in response if r == mask_id)
        if num_masked == 0:
            break

        decode_indices = None
        if algorithm == "low-confidence":
            # 取每个位置的置信度（softmax 最大概率），优先解码高置信位置
            probs = torch.softmax(logits[0].float(), dim=-1)
            confidences, _ = probs.max(dim=-1)
            num_to_decode = max(1, round(num_masked * (1.0 - s / t)))
            masked_conf = [
                (i, confidences[prompt_len + i].item())
                for i in range(answer_length) if response[i] == mask_id
            ]
            masked_conf.sort(key=lambda x: x[1], reverse=True)
            decode_indices = {i for i, _ in masked_conf[:num_to_decode]}

        # correction 视图：已确认 token + 掩码位置的当前预测
        correction_view = [
            response[i] if response[i] != mask_id else pred_ids[i]
            for i in range(answer_length)
        ]
        num_changed = sum(
            1 for i in range(answer_length)
            if response[i] == mask_id and correction_view[i] != prev_correction[i]
        )
        prev_correction = correction_view.copy()

        # 应用掩码决策
        prev_response = response
        if algorithm == "low-confidence":
            num_to_decode_final = len(decode_indices)
            for i in range(answer_length):
                if prev_response[i] != mask_id:
                    pred_ids[i] = prev_response[i]
                elif i not in decode_indices:
                    pred_ids[i] = mask_id
        else:  # random remasking：以 s/t 概率重新掩码
            num_to_decode_final = 0
            for i in range(answer_length):
                if prev_response[i] != mask_id:
                    pred_ids[i] = prev_response[i]
                elif random.random() < s / t:
                    pred_ids[i] = mask_id
                else:
                    num_to_decode_final += 1

        response = pred_ids
        new_tokens = {
            i for i in range(answer_length)
            if prev_response[i] == mask_id and response[i] != mask_id
        }

        yield pack(step + 1, new_tokens, num_to_decode_final, num_changed)
        time.sleep(0.04)


# --------------------------------------------------------------------------
# HTML 渲染：把 token 序列渲染为带动画的现代化面板
# --------------------------------------------------------------------------
def render_tokens(state, show_mode):
    tokenizer = get_tokenizer()
    mask_id = tokenizer.mask_token_id
    pad_token = tokenizer.pad_token or ""
    response, correction_view, new_tokens = (
        state["response"], state["correction_view"], state["new_tokens"],
    )

    # 为每个位置分配样式类
    classes, ids = [], []
    for i, r in enumerate(response):
        if show_mode == "correction":
            if r != mask_id:
                classes.append("done"); ids.append(r)
            else:
                classes.append("pred"); ids.append(correction_view[i])
        else:
            if r == mask_id:
                classes.append("mask"); ids.append(None)
            elif i in new_tokens:
                classes.append("new"); ids.append(r)
            else:
                classes.append("done"); ids.append(r)

    # 连续同类分组后整体 decode，保证 BPE 文本正确
    parts, i, n = [], 0, len(response)
    while i < n:
        j = i
        while j + 1 < n and classes[j + 1] == classes[i]:
            j += 1
        if classes[i] == "mask":
            parts.append('<span class="mask"></span>' * (j - i + 1))
        else:
            text = tokenizer.decode([t for t in ids[i:j + 1]])
            if pad_token:
                text = text.replace(pad_token, " ")
            parts.append(f'<span class="{classes[i]}">{html.escape(text)}</span>')
        i = j + 1
    return "".join(parts)


def render_panel(prompt_text, state, total_steps, show_mode, token_count=None):
    total = len(state["response"])
    decoded = total - state["num_masked"]
    pct = decoded / max(total, 1) * 100
    tokens_html = render_tokens(state, show_mode)

    changed_chip = (
        f'<span class="chip">修正 <b>{state["num_changed"]}</b></span>'
        if show_mode == "correction" else ""
    )
    if token_count is None:
        label = "PROMPT"
    elif token_count > PROMPT_LEN:
        label = f"PROMPT · {token_count}→{PROMPT_LEN}"
    else:
        label = f"PROMPT · {PROMPT_LEN}/{PROMPT_LEN}"
    return f"""
    <div class="panel">
      <div class="prompt-line"><span class="label">{label}</span>{html.escape(prompt_text)}</div>
      <div class="progress"><div class="bar" style="width:{pct:.1f}%"></div></div>
      <div class="tokens">{tokens_html}</div>
      <div class="stats">
        <span class="chip">步数 <b>{state['step']}/{total_steps}</b></span>
        <span class="chip">已解码 <b>{decoded}/{total}</b></span>
        <span class="chip">本步新增 <b>{state['num_to_decode']}</b></span>
        {changed_chip}
      </div>
    </div>
    """


def placeholder_panel(message, warn=False):
    cls = "placeholder-text warn" if warn else "placeholder-text"
    return (
        '<div class="panel placeholder">'
        f'<div class="{cls}">{html.escape(message)}</div></div>'
    )


EMPTY_PANEL = placeholder_panel("设置参数并点击「开始生成」，观看扩散模型逐步去掩码生成文本")

# --------------------------------------------------------------------------
# Gradio 事件处理
# --------------------------------------------------------------------------
def generate(checkpoint, prompt_text, algorithm, show_mode, steps):
    if not prompt_text or not prompt_text.strip():
        yield placeholder_panel("请输入 Prompt", warn=True), ""
        return

    tokenizer = get_tokenizer()
    model = get_model(checkpoint)

    raw_ids = tokenizer(prompt_text.strip(), add_special_tokens=True)["input_ids"]
    # 训练时位置 0~PROMPT_LEN-1 从不被掩码且总是真实文本，因此推理 prompt
    # 必须恰好是 PROMPT_LEN 个真实 token：超长强行截断，不足则拒绝生成。
    if len(raw_ids) < PROMPT_LEN:
        yield placeholder_panel(
            f"Prompt 太短：共 {len(raw_ids)} 个 token，至少需要 {PROMPT_LEN} 个。"
            "请补充内容，或点击「随机数据集示例」。",
            warn=True,
        ), ""
        return
    token_count = len(raw_ids)
    prompt_ids = raw_ids[:PROMPT_LEN]

    last_state = None
    for state in stream_inference(model, prompt_ids, algorithm, int(steps)):
        last_state = state
        yield render_panel(prompt_text, state, int(steps), show_mode, token_count), ""

    final_text = (
        tokenizer.decode(last_state["response"], skip_special_tokens=True).strip()
        if last_state else ""
    )
    yield render_panel(prompt_text, last_state, int(steps), show_mode, token_count), final_text


def random_example():
    ds = get_dataset()
    tokenizer = get_tokenizer()
    # 训练只用了数据集前 10000 条，演示也只从这部分取样
    sample = ds[random.randrange(min(10000, len(ds)))]["input_ids"].tolist()
    ids = [i for i in sample[:PROMPT_LEN] if i != tokenizer.pad_token_id]
    return tokenizer.decode(ids).strip()


# --------------------------------------------------------------------------
# 界面
# --------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container { max-width: 1100px !important; }
.llada-header { text-align: center; padding: 18px 0 6px; }
.llada-header h1 {
    font-size: 30px; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #6366f1, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.llada-header p { color: #8b93a7; font-size: 14px; margin: 6px 0 0; }
.panel {
    background: #0f1420; border: 1px solid #232b3d; border-radius: 14px;
    padding: 20px 22px; min-height: 300px;
}
.panel.placeholder { display: flex; align-items: center; justify-content: center; }
.placeholder-text { color: #4b5568; font-size: 14px; }
.placeholder-text.warn { color: #f59e0b; }
.prompt-line { color: #c3cad9; font-size: 14px; margin-bottom: 14px; line-height: 1.6; }
.prompt-line .label {
    background: #1c2436; color: #7dd3fc; border-radius: 6px;
    padding: 2px 8px; font-size: 11px; font-weight: 700; margin-right: 10px;
    letter-spacing: 1px;
}
.progress { height: 6px; background: #1c2436; border-radius: 3px; overflow: hidden; margin-bottom: 18px; }
.bar { height: 100%; background: linear-gradient(90deg, #6366f1, #22d3ee); transition: width .15s ease; }
.tokens {
    font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 15px;
    line-height: 2.1; white-space: pre-wrap; word-break: break-word; color: #e6eaf2;
    min-height: 120px;
}
.mask {
    display: inline-block; width: 0.85em; height: 1.05em; background: #2a3350;
    border-radius: 4px; margin: 0 1.5px; vertical-align: -0.18em;
    animation: pulse 1.1s ease-in-out infinite;
}
@keyframes pulse { 0%, 100% { opacity: .4; } 50% { opacity: 1; } }
.new  { background: rgba(34, 197, 94, .16); color: #86efac; border-radius: 4px; }
.pred { color: #7dd3fc; opacity: .7; font-style: italic; }
.done { color: #e6eaf2; }
.stats { margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap; }
.chip { background: #1c2436; color: #8b93a7; border-radius: 999px; padding: 4px 12px; font-size: 12px; }
.chip b { color: #e6eaf2; }
"""


def build_app():
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")))
    default_ckpt = os.path.join(CHECKPOINT_DIR, "final_model.pth")
    if default_ckpt not in checkpoints and checkpoints:
        default_ckpt = checkpoints[0]

    with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS, title="LLaDA 扩散文本生成") as demo:
        gr.HTML("""
        <div class="llada-header">
          <h1>LLaDA 扩散语言模型</h1>
          <p>从全掩码开始，迭代去噪生成文本 · Diffusion Language Model Demo</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                ckpt = gr.Dropdown(
                    choices=checkpoints, value=default_ckpt, label="模型权重",
                )
                prompt = gr.Textbox(
                    label="Prompt（至少 32 token，超长将截断；TinyStories 风格效果更好）",
                    value="Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her best friend. One sunny day, they found a big red ball near the slide.",
                    lines=3,
                )
                rand_btn = gr.Button("随机数据集示例", size="sm")
                algo = gr.Radio(
                    choices=[("低置信度重掩码（高质量）", "low-confidence"),
                             ("随机重掩码（基线）", "random")],
                    value="low-confidence", label="推理算法",
                )
                mode = gr.Radio(
                    choices=[("decoding · 逐步去掩码", "decoding"),
                             ("correction · 观察模型修正预测", "correction")],
                    value="decoding", label="显示模式",
                )
                steps = gr.Slider(10, 256, value=100, step=1, label="采样步数")
                go = gr.Button("开始生成", variant="primary", size="lg")

            with gr.Column(scale=2):
                panel = gr.HTML(EMPTY_PANEL)
                final = gr.Textbox(label="最终生成结果", lines=4, interactive=False)

        go.click(
            generate,
            inputs=[ckpt, prompt, algo, mode, steps],
            outputs=[panel, final],
            show_progress="hidden",
        )
        rand_btn.click(random_example, outputs=prompt)

    return demo


if __name__ == "__main__":
    validate_config()
    app = build_app()
    app.launch()
