import math
import io
import base64
import os
from flask import Flask, request, render_template_string
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, CSS4_COLORS

app = Flask(__name__)

# ── Cores ─────────────────────────────────────────────────────────────────────
CORES_PT = {
    "vermelho": "red", "azul": "blue", "branco": "white", "preto": "black",
    "verde": "green", "amarelo": "yellow", "rosa": "pink", "roxo": "purple",
    "laranja": "orange", "cinza": "gray", "marrom": "brown", "bege": "beige",
    "creme": "ivory", "turquesa": "turquoise", "lilas": "plum", "lilás": "plum",
    "azul marinho": "navy", "azul claro": "skyblue", "verde claro": "lightgreen",
    "verde escuro": "darkgreen", "verde militar": "#4a5e3a",
    "vermelho escuro": "darkred", "dourado": "gold", "prata": "silver",
    "salmon": "salmon", "coral": "coral", "magenta": "magenta", "cyan": "cyan",
    "cafe com leite": "#c4a882", "café com leite": "#c4a882",
    "cafe": "#6f4e37", "café": "#6f4e37",
    "telha": "#b5651d", "terracota": "#c0533a",
    "areia": "#c2b280", "off-white": "#f5f5dc",
}

def resolve_color(name):
    n = name.strip().lower()
    if n in CORES_PT:
        return CORES_PT[n]
    if n in CSS4_COLORS:
        return n
    try:
        to_rgb(n)
        return n
    except Exception:
        return None

def cycle_length(pattern):
    return sum(c for _, c in pattern)

def expand_full_cycles(pattern, target):
    cl = cycle_length(pattern)
    n = math.ceil(target / cl)
    out = []
    for _ in range(n):
        for c, cnt in pattern:
            out += [c] * cnt
    return out

def make_warp(pattern, target_n):
    colors = expand_full_cycles(pattern, target_n)
    colors[-1] = pattern[0][0]
    return colors

def parse_pattern_str(s):
    items = [x.strip() for x in s.split(",") if x.strip()]
    result = []
    for item in items:
        parts = item.strip().split(None, 1)
        if len(parts) < 2:
            return None, f"Formato inválido: '{item}' (use: quantidade cor)"
        try:
            count = int(parts[0])
        except ValueError:
            return None, f"Quantidade inválida: '{parts[0]}'"
        color = resolve_color(parts[1])
        if not color:
            return None, f"Cor não reconhecida: '{parts[1]}'"
        result.append((color, count))
    return result, None

def generate_image(warp_w_cm, warp_dens, warp_pat,
                   sections_def, weft_dens, target_section_cm=10):
    target_n_warp = warp_w_cm * warp_dens
    warp_colors = make_warp(warp_pat, target_n_warp)
    n_warp = len(warp_colors)
    real_warp_cm = n_warp / warp_dens

    target_passes = int(target_section_cm * weft_dens)
    weft_colors = []
    boundaries = [0]
    for pat in sections_def:
        seg = expand_full_cycles(pat, target_passes)
        weft_colors += seg
        boundaries.append(len(weft_colors))

    n_weft = len(weft_colors)
    real_weft_cm = n_weft / weft_dens

    max_dim = max(n_warp, n_weft)
    ppt = max(4, min(10, int(700 / max_dim)))

    img = np.zeros((n_weft, n_warp, 3), dtype=np.float32)
    for j in range(n_weft):
        for i in range(n_warp):
            if (i + j) % 2 == 0:
                img[j, i] = to_rgb(warp_colors[i])
            else:
                img[j, i] = to_rgb(weft_colors[j])

    img_s = np.repeat(np.repeat(img, ppt, axis=0), ppt, axis=1)

    fig_w = max(n_warp * ppt / 100 + 2.5, 5)
    fig_h = max(n_weft * ppt / 100 + 2.5, 6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
    ax.imshow(img_s, origin="upper", aspect="equal", interpolation="nearest")

    cx = np.arange(0, real_warp_cm + 0.01, 2)
    ax.set_xticks(cx * warp_dens * ppt - 0.5)
    ax.set_xticklabels([f"{v:.0f}" for v in cx], fontsize=6)
    ax.set_xlabel("Largura (cm)", fontsize=8)

    cy = np.arange(0, real_weft_cm + 0.01, 5)
    ax.set_yticks(cy * weft_dens * ppt - 0.5)
    ax.set_yticklabels([f"{v:.0f}" for v in cy], fontsize=6)
    ax.set_ylabel("Comprimento (cm)", fontsize=8)

    for idx in range(len(sections_def)):
        sp = boundaries[idx]
        ep = boundaries[idx + 1]
        mid = (sp + ep) / 2 * ppt
        if idx > 0:
            ax.axhline(y=sp * ppt - 0.5, color="white", lw=1.2, ls="--", alpha=0.8)
        cm_s = boundaries[idx] / weft_dens
        cm_e = boundaries[idx + 1] / weft_dens
        ax.text(n_warp * ppt + 4, mid,
                f"Seção {idx+1}\n({cm_s:.0f}–{cm_e:.0f}cm)",
                va="center", ha="left", fontsize=6, color="#222",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0))

    ax.set_title(
        f"Simulação — Tear de Pente Liço\n"
        f"Urdume: {real_warp_cm:.0f}cm × {warp_dens} fios/cm  |  "
        f"Trama: {real_weft_cm:.0f}cm × {weft_dens} pass/cm",
        fontsize=9, pad=6)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode(), real_warp_cm, real_weft_cm

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML_FORM = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🧵 Simulador de Tear de Pente Liço</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #f5f0eb; color: #2c2c2c; padding: 16px; }
  h1 { font-size: 1.3rem; color: #4a5e3a; margin-bottom: 4px; }
  .subtitle { font-size: 0.85rem; color: #777; margin-bottom: 20px; }
  .card { background: white; border-radius: 12px; padding: 18px; margin-bottom: 16px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  .card h2 { font-size: 1rem; color: #4a5e3a; margin-bottom: 12px; }
  label { display: block; font-size: 0.82rem; font-weight: 600; color: #555;
          margin-bottom: 4px; margin-top: 10px; }
  input, select {
    width: 100%; padding: 10px 12px; border: 1.5px solid #ddd; border-radius: 8px;
    font-size: 0.9rem; background: #fafafa; transition: border 0.2s;
  }
  input:focus, select:focus { outline: none; border-color: #4a5e3a; background: white; }
  .hint { font-size: 0.75rem; color: #999; margin-top: 3px; }
  .section-block { background: #f9f6f2; border: 1px solid #e8e0d5; border-radius: 8px;
                   padding: 12px; margin-top: 8px; }
  .add-btn { padding: 7px 14px; border: none; border-radius: 7px; cursor: pointer;
             font-size: 0.82rem; font-weight: 600; margin-top: 8px;
             background: #e8f0e8; color: #4a5e3a; }
  .rem-btn { padding: 5px 10px; border: none; border-radius: 7px; cursor: pointer;
             font-size: 0.78rem; font-weight: 600; margin-left: 6px;
             background: #fde8e8; color: #c0392b; }
  .submit-btn { width: 100%; padding: 14px; background: #4a5e3a; color: white;
                border: none; border-radius: 10px; font-size: 1rem; font-weight: 700;
                cursor: pointer; margin-top: 8px; }
  .submit-btn:hover { background: #3a4e2a; }
  .cores-ref { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .cor-chip { display: flex; align-items: center; gap: 5px; background: #f0f0f0;
              border-radius: 20px; padding: 3px 10px; font-size: 0.75rem; }
  .cor-dot { width: 14px; height: 14px; border-radius: 50%; border: 1px solid #ccc; flex-shrink: 0; }
</style>
</head>
<body>
<h1>🧵 Simulador de Tear de Pente Liço</h1>
<p class="subtitle">Configure sua peça e visualize antes de tecer!</p>
<form method="POST" action="/simular">
  <div class="card">
    <h2>📐 Urdume (fios verticais)</h2>
    <label>Largura desejada (cm)</label>
    <input type="number" name="warp_width" value="40" min="5" max="200" step="0.5" required>
    <p class="hint">Ajustado automaticamente para ciclo completo.</p>
    <label>Densidade</label>
    <select name="warp_density">
      <option value="2">2 fios/cm (barbante grosso, n°6)</option>
      <option value="4">4 fios/cm (linha fina)</option>
    </select>
    <label>Padrão de cores</label>
    <input type="text" name="warp_pattern" value="1 verde militar, 2 café com leite"
           placeholder="ex: 1 verde militar, 2 café com leite" required>
    <p class="hint">Formato: <b>quantidade cor</b>, separados por vírgula. Começa e termina na 1ª cor.</p>
  </div>
  <div class="card">
    <h2>🧶 Trama (passadas horizontais)</h2>
    <label>Densidade</label>
    <select name="weft_density">
      <option value="2">2 passadas/cm (barbante grosso, n°6)</option>
      <option value="4">4 passadas/cm (linha fina)</option>
    </select>
    <label>Comprimento alvo por seção (cm)</label>
    <input type="number" name="section_cm" value="10" min="2" max="100" step="0.5">
    <p class="hint">Arredondado para ciclo completo automaticamente.</p>
    <p style="margin-top:14px;font-size:0.85rem;font-weight:600;color:#555;">Seções da trama:</p>
    <div id="sections-container">
      <div class="section-block" data-idx="0">
        <label>Seção 1</label>
        <input type="text" name="section_0" value="1 café com leite"
               placeholder="ex: 1 café com leite">
      </div>
    </div>
    <button type="button" class="add-btn" onclick="addSection()">+ Adicionar seção</button>
  </div>
  <div class="card">
    <h2>🎨 Cores disponíveis</h2>
    <div class="cores-ref">
      <span class="cor-chip"><span class="cor-dot" style="background:#4a5e3a"></span>verde militar</span>
      <span class="cor-chip"><span class="cor-dot" style="background:#c4a882"></span>café com leite</span>
      <span class="cor-chip"><span class="cor-dot" style="background:#b5651d"></span>telha</span>
      <span class="cor-chip"><span class="cor-dot" style="background:navy"></span>azul marinho</span>
      <span class="cor-chip"><span class="cor-dot" style="background:white;border:1px solid #ccc"></span>branco</span>
      <span class="cor-chip"><span class="cor-dot" style="background:black"></span>preto</span>
      <span class="cor-chip"><span class="cor-dot" style="background:red"></span>vermelho</span>
      <span class="cor-chip"><span class="cor-dot" style="background:#6f4e37"></span>café</span>
      <span class="cor-chip"><span class="cor-dot" style="background:gray"></span>cinza</span>
      <span class="cor-chip"><span class="cor-dot" style="background:pink"></span>rosa</span>
      <span class="cor-chip"><span class="cor-dot" style="background:yellow"></span>amarelo</span>
      <span class="cor-chip"><span class="cor-dot" style="background:orange"></span>laranja</span>
      <span class="cor-chip"><span class="cor-dot" style="background:gold"></span>dourado</span>
      <span class="cor-chip"><span class="cor-dot" style="background:turquoise"></span>turquesa</span>
      <span class="cor-chip"><span class="cor-dot" style="background:purple"></span>roxo</span>
      <span class="cor-chip"><span class="cor-dot" style="background:brown"></span>marrom</span>
      <span class="cor-chip"><span class="cor-dot" style="background:beige;border:1px solid #ccc"></span>bege</span>
      <span class="cor-chip"><span class="cor-dot" style="background:#c0533a"></span>terracota</span>
    </div>
    <p class="hint" style="margin-top:8px">Também aceita inglês e hex (#RRGGBB).</p>
  </div>
  <button type="submit" class="submit-btn">✨ Gerar simulação</button>
</form>
<script>
let sectionCount = 1;
function addSection() {
  const container = document.getElementById('sections-container');
  const idx = sectionCount;
  const div = document.createElement('div');
  div.className = 'section-block';
  div.dataset.idx = idx;
  div.innerHTML = `<label>Seção ${idx+1}
    <button type="button" class="rem-btn" onclick="removeSection(this)">✕</button></label>
    <input type="text" name="section_${idx}" placeholder="ex: 2 verde militar, 3 café com leite">`;
  container.appendChild(div);
  sectionCount++;
  renumber();
}
function removeSection(btn) {
  btn.closest('.section-block').remove();
  renumber();
}
function renumber() {
  const blocks = document.querySelectorAll('.section-block');
  blocks.forEach((b, i) => {
    b.querySelector('input').name = `section_${i}`;
    b.querySelector('label').childNodes[0].textContent = `Seção ${i+1} `;
  });
  sectionCount = blocks.length;
}
</script>
</body>
</html>"""

@app.route("/", methods=["GET"])
def index():
    return HTML_FORM

@app.route("/simular", methods=["POST"])
def simular():
    try:
        warp_w  = float(request.form.get("warp_width", "40").replace(",", "."))
        warp_d  = int(request.form.get("warp_density", "2"))
        weft_d  = int(request.form.get("weft_density", "2"))
        sec_cm  = float(request.form.get("section_cm", "10").replace(",", "."))

        warp_pat, err = parse_pattern_str(request.form.get("warp_pattern", ""))
        if err:
            raise ValueError(f"Urdume — {err}")

        sections = []
        i = 0
        while True:
            raw = request.form.get(f"section_{i}", "").strip()
            if not raw and i > 0:
                break
            if raw:
                pat, err = parse_pattern_str(raw)
                if err:
                    raise ValueError(f"Seção {i+1} — {err}")
                sections.append(pat)
            elif i == 0:
                raise ValueError("Adicione pelo menos uma seção na trama.")
            i += 1
            if i > 50:
                break

        img_b64, rw, rl = generate_image(warp_w, warp_d, warp_pat, sections, weft_d, sec_cm)

        html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>🧵 Resultado</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',sans-serif; background:#f5f0eb; color:#2c2c2c; padding:16px; }}
  h1 {{ font-size:1.3rem; color:#4a5e3a; margin-bottom:4px; }}
  .subtitle {{ font-size:0.85rem; color:#777; margin-bottom:16px; }}
  .card {{ background:white; border-radius:12px; padding:16px; margin-bottom:16px;
           box-shadow:0 2px 8px rgba(0,0,0,0.08); }}
  .info-box {{ background:#e8f0e8; border-radius:8px; padding:12px;
               margin-bottom:12px; font-size:0.85rem; color:#2c4a2c; }}
  .result-img {{ width:100%; border-radius:10px; box-shadow:0 2px 12px rgba(0,0,0,0.12); }}
  .btn {{ display:block; text-align:center; padding:13px; border-radius:9px;
          text-decoration:none; font-weight:700; margin-top:10px; font-size:0.95rem; }}
  .btn-back {{ background:#f5f0eb; color:#4a5e3a; border:2px solid #4a5e3a; }}
  .btn-dl {{ background:#4a5e3a; color:white; }}
</style>
</head>
<body>
<h1>🧵 Sua simulação</h1>
<p class="subtitle">Tear de Pente Liço — resultado</p>
<div class="info-box">
  📐 Largura real: <b>{rw:.1f} cm</b> &nbsp;|&nbsp;
  📏 Comprimento real: <b>{rl:.1f} cm</b><br>
  <small>Dimensões ajustadas para ciclos completos.</small>
</div>
<div class="card">
  <img src="data:image/png;base64,{img_b64}" class="result-img" alt="Simulação">
</div>
<a href="data:image/png;base64,{img_b64}" download="simulacao_tear.png" class="btn btn-dl">
  ⬇️ Baixar imagem PNG
</a>
<a href="/" class="btn btn-back">← Nova simulação</a>
</body>
</html>"""
        return html

    except Exception as e:
        return f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Erro</title>
<style>body{{font-family:sans-serif;padding:20px;background:#f5f0eb}}
.error{{background:#fde8e8;color:#c0392b;padding:16px;border-radius:10px;margin-bottom:16px}}
a{{display:block;text-align:center;padding:12px;background:#4a5e3a;color:white;
border-radius:8px;text-decoration:none;font-weight:700;margin-top:12px}}</style></head>
<body><div class="error">❌ {str(e)}</div>
<a href="/">← Voltar e corrigir</a></body></html>"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
