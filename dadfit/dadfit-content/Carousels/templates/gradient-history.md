# Gradient Background History

Four states tried on batch 1 carousels (May 2026).

---

## v0 — No Gradient (flat)

All slides used a flat `background: #1E1E1E` — the base `--primary-bg` CSS variable. No linear gradient, no radial glow.

**Feedback**: Looked clean but too flat / generic. Lacked visual depth and slide-type differentiation.

---

## v1 — High Intensity (original)

Color stops pushed far from `#1E1E1E` to create strong color-family tinting. Radial glow opacities 0.08–0.10 on A/G, 0.05–0.06 on B/D/F, 0.04–0.06 on C, 0.05 on E, 0.06–0.08 on H.

**Feedback**: Too vibrant / heavy on screen and in exported PNGs.

Sample values:
```
A:  linear-gradient(145deg,#0c1610 0%,#121e15 30%,#17211a 58%,#1c1e1c 100%)
    radial @ 75% 88%: rgba(52,195,99,0.10)
C:  linear-gradient(145deg,#1c1212 0%,#201616 35%,#1f1b1b 65%,#1E1E1E 100%)
    radial: rgba(255,107,107,0.06)
H:  linear-gradient(145deg,#060c08→#0c120d→#111512)   ← darkened ~40% mid-iteration
    radial: rgba(52,195,99,0.08)
```

---

## v2 — CTA Darkened (partial tweak)

Only the H family (CTA slides H1–H4) was darkened by ~40% while all other families stayed at v1 intensity. Reason: CTA slides felt too bright compared to content slides.

```
H:  linear-gradient(145deg,#060c08 0%,#090f09 25%,#0c120d 50%,#0f1410 75%,#111512 100%)
    radial: rgba(52,195,99,0.06–0.08)
```

All other families unchanged from v1.

---

## v3 — Half Intensity (current, in all files)

All color stops moved ~50% closer to flat `#1E1E1E`. Radial glow opacities halved across every family. This is what's baked into all carousels and snippets today.

**Feedback**: Subtle, still distinct per slide-type family.

Sample values:
```
A:  linear-gradient(145deg,#151a17 0%,#181e19 30%,#1a1f1c 58%,#1c1e1c 100%)
    radial: rgba(52,195,99,0.05)
B:  linear-gradient(150deg,#18191f 0%,#1a1c22 35%,#1c1e21 60%,#1c1e22 100%)
    radial: rgba(52,195,99,0.03)
C:  linear-gradient(145deg,#1d1818 0%,#1f1a1a 35%,#1e1c1c 65%,#1E1E1E 100%)
    radial: rgba(255,107,107,0.03)
D:  linear-gradient(145deg,#171c1d 0%,#191e1f 30%,#1b1e1e 58%,#1c1e1e 100%)
    radial: rgba(52,195,99,0.04)
E:  linear-gradient(145deg,#1c1917 0%,#1e1b17 35%,#1d1c19 60%,#1e1e1c 100%)
    radial: rgba(255,180,80,0.03)
F:  linear-gradient(150deg,#181a19 0%,#1a1c1b 35%,#1b1d1c 60%,#1e1e1e 100%)
    radial: rgba(52,195,99,0.03)
G:  linear-gradient(145deg,#161a17 0%,#191e19 35%,#1b1f1b 60%,#1c1e1c 100%)
    radial: rgba(52,195,99,0.04)
H:  linear-gradient(145deg,#121513 0%,#131613 25%,#151815 50%,#161917 75%,#171918 100%)
    radial: rgba(52,195,99,0.03–0.04)
```

Full values always live in `patch_gradients.py` → `GRADIENTS` dict.

---

## Slide families skipped (photo background)

A3, B3, D2, F1 — these use a full-bleed photo so no gradient is applied.
