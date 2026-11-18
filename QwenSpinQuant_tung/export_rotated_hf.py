#!/usr/bin/env python3
"""Gop (fuse) rotation R1 + R2 cua SpinQuant vao trong so Qwen2 va luu thanh model
HuggingFace CHUAN (bf16/fp16), de convert sang GGUF bang llama.cpp.

Khac voi ptq.py / ptq_model:
  - KHONG ap R4 (Hadamard online truoc down_proj). R4 phai chay luc inference, llama.cpp
    khong co. Bo R4 thi model xoay van TUONG DUONG model goc (chi R1, R2 duoc gop vao trong so).
  - KHONG add_actquant, KHONG quantize (ca GPTQ lan RTN): trong so van la so thuc.
    Viec quantize se do llama-quantize lam.
  - Vi khong con Hadamard cho down_proj nen KHONG CAN pad MLP: giu nguyen intermediate_size
    (18944 voi Qwen2.5-7B). Neu R.bin cua ban duoc train tren model da pad thi van dung duoc,
    vi R1 (hidden x hidden) va R2 (head_dim x head_dim) khong phu thuoc intermediate_size.
  - Khong dung R3 (chi duoc them khi k_bits < 16, cau hinh cua ban la k_bits 16).

Chay trong thu muc QwenSpinQuant (can eval_utils/, utils/ cua repo), tren GPU con it nhat
~10 GB trong (embedding/lm_head duoc nhan o float64 tren GPU).

Vi du:
  python export_rotated_hf.py \
    --input_model /path/Qwen2.5-7B-Instruct \
    --rotation_path rotation/R.bin \
    --output_dir /path/Qwen2.5-7B-Instruct-rot \
    --verify
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

FALLBACK_TEXT = (
    "The history of computing is closely tied to the history of mathematics. Early mechanical "
    "calculators were built to reduce errors in navigation tables, and later machines were "
    "designed to follow sequences of instructions stored in memory. In the twentieth century, "
    "advances in electronics made it possible to build computers that were both faster and "
    "smaller. Today, large language models are trained on text collected from many sources, and "
    "they can be compressed with quantization so that they run on devices with limited memory. "
    "A river that crosses several countries often becomes a subject of treaties, because the "
    "water it carries is used for farming, drinking and electricity. Bread is made from flour, "
    "water, salt and yeast, but small changes in temperature or time can change its texture."
)


def get_dtype(name):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


# --------------------------------------------------------------------- xoay
def rotate_fuse_r1_r2(model, rotation_path):
    """Giong rotation_utils.rotate_model nhung bo buoc ap Hadamard (R4) len down_proj."""
    import torch
    import tqdm
    from eval_utils import rotation_utils
    from utils import utils

    R = torch.load(rotation_path, map_location="cpu")
    cfg = model.config
    num_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // num_heads
    if tuple(R["R1"].shape) != (cfg.hidden_size, cfg.hidden_size):
        raise ValueError(
            f"R1 co shape {tuple(R['R1'].shape)} nhung hidden_size cua model la {cfg.hidden_size}: "
            "R.bin khong phai cua model nay."
        )

    with torch.inference_mode():
        R1 = R["R1"].cuda().to(torch.float64)
        rotation_utils.rotate_embeddings(model, R1)
        rotation_utils.rotate_head(model, R1)
        utils.cleanup_memory()

        for idx, layer in enumerate(tqdm.tqdm(model.model.layers, unit="layer", desc="Rotating (R1,R2)")):
            R2 = R[f"model.layers.{idx}.self_attn.R2"].cuda().to(torch.float64)
            rotation_utils.rotate_attention_inputs(layer, R1)
            rotation_utils.rotate_attention_output(layer, R1)
            rotation_utils.rotate_mlp_input(layer, R1)

            # down_proj: chi xoay dau ra bang R1. KHONG goi apply_exact_had_to_linear (R4).
            W = layer.mlp.down_proj
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
            W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(device="cuda", dtype=torch.float64)
                W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)

            rotation_utils.rotate_ov_proj(layer, num_heads, head_dim, R2=R2)


def check_norms_fused(model):
    """Sau fuse, moi RMSNorm phai co weight = 1. Neu khong, phep xoay R1 se SAI."""
    import torch

    found, bad = 0, 0
    for name, p in model.named_parameters():
        if "norm" in name and name.endswith(".weight"):
            found += 1
            if not torch.allclose(p.float(), torch.ones_like(p, dtype=torch.float32)):
                bad += 1
    expected = 2 * model.config.num_hidden_layers + 1
    if found != expected:
        raise RuntimeError(f"Tim thay {found} weight norm, ky vong {expected}. fuse_layer_norms da doi cau truc model?")
    if bad:
        raise RuntimeError(f"{bad}/{found} weight norm khac 1 sau fuse -> rotation khong tuong duong, dung export.")
    print(f"[ok] {found} RMSNorm deu co weight = 1 sau fuse")


# ------------------------------------------------------------------- export
def export(args):
    import torch
    import transformers
    from eval_utils.modeling_qwen2 import Qwen2ForCausalLM
    from utils import fuse_norm_utils

    config = transformers.AutoConfig.from_pretrained(args.input_model)
    tie = bool(config.tie_word_embeddings)
    if tie:  # SpinQuant khong tuong thich tie_word_embeddings: tach lm_head ra
        config.tie_word_embeddings = False
    model = Qwen2ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.input_model,
        config=config,
        torch_dtype=get_dtype(args.load_dtype),
    )
    if tie:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.eval()

    fuse_norm_utils.fuse_layer_norms(model)
    check_norms_fused(model)
    rotate_fuse_r1_r2(model, args.rotation_path)

    model.to(get_dtype(args.dtype))
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    transformers.AutoTokenizer.from_pretrained(args.input_model).save_pretrained(args.output_dir)
    for extra in ("generation_config.json",):
        src = os.path.join(args.input_model, extra)
        if os.path.isfile(src):
            import shutil

            shutil.copy(src, os.path.join(args.output_dir, extra))
    print(f"[ok] da luu model xoay (R1,R2 gop vao trong so, khong R4) -> {args.output_dir}")
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ------------------------------------------------------------------- verify
def load_text(tokenizer, n_tokens):
    try:
        import datasets

        ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(t for t in ds["text"][:400] if t.strip())
    except Exception:
        text = FALLBACK_TEXT
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :n_tokens]
    return ids


def verify(args):
    """So logits cua model goc va model da xoay (fp, chua quantize). Phai gan nhu giong nhau."""
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM

    dev = "cuda"
    tok = transformers.AutoTokenizer.from_pretrained(args.input_model)
    ids = load_text(tok, args.verify_tokens)
    print(f"[verify] {ids.shape[1]} tokens")

    def logits_of(path):
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=get_dtype(args.dtype)).to(dev).eval()
        with torch.inference_mode():
            lg = m(ids.to(dev)).logits[0].float().cpu()
        del m
        gc.collect()
        torch.cuda.empty_cache()
        return lg

    a = logits_of(args.input_model)
    b = logits_of(args.output_dir)
    target = ids[0, 1:]
    nll_a = F.cross_entropy(a[:-1], target).item()
    nll_b = F.cross_entropy(b[:-1], target).item()
    la, lb = F.log_softmax(a, -1), F.log_softmax(b, -1)
    kl = (la.exp() * (la - lb)).sum(-1).mean().item()
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"[verify] NLL goc      : {nll_a:.4f}")
    print(f"[verify] NLL da xoay  : {nll_b:.4f}   (chenh {abs(nll_a - nll_b):.4f})")
    print(f"[verify] KL(goc||xoay): {kl:.5f}")
    print(f"[verify] top-1 trung  : {top1 * 100:.1f}%")
    print("Neu rotation dung: NLL gan nhu bang nhau, KL rat nho, top-1 gan 100%. "
          "Chenh lech nho la do lam tron bf16. KL lon hon nhieu (vd > 0.1) hoac NLL khac ro rang = co loi.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_model", required=True)
    ap.add_argument("--rotation_path", required=True, help="R.bin tu optimize_rotation.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="dtype luu ra")
    ap.add_argument("--load_dtype", default="fp32", choices=["fp32", "bf16"],
                    help="dtype khi xoay (fp32 chinh xac hon, can ~30 GB RAM voi 7B; bf16 can ~15 GB)")
    ap.add_argument("--spinquant_dir", default=os.environ.get("QWEN_SPINQUANT_DIR", os.getcwd()))
    ap.add_argument("--verify", action="store_true", help="so logits model goc va model xoay sau khi luu")
    ap.add_argument("--verify_tokens", type=int, default=512)
    ap.add_argument("--skip_export", action="store_true", help="chi chay --verify tren output_dir da co")
    args = ap.parse_args()

    args.input_model = os.path.abspath(args.input_model)
    args.rotation_path = os.path.abspath(args.rotation_path)
    args.output_dir = os.path.abspath(args.output_dir)
    sys.path.insert(0, os.path.abspath(args.spinquant_dir))

    if not os.path.isfile(os.path.join(args.input_model, "config.json")):
        sys.exit(f"Khong thay config.json trong {args.input_model}")
    if not os.path.isfile(args.rotation_path):
        sys.exit(f"Khong thay {args.rotation_path}")

    if not args.skip_export:
        export(args)
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
