# 📈 Approx Borrow Size Analysis

A powerful and intuitive **visual analytics suite** for exploring  
the relationship between **Price** and **Approx Borrow Size** in Thinkorswim.

This project combines:
- A robust Python backend for data parsing and visualization, and  
- A modern Tkinter GUI launcher for quick configuration and presets.

---

## 🧭 Overview

**Approx Borrow Size Analysis** loads multiple CSV snapshots exported from your trading watchlist in Thinkorswim, merges them by timestamp, and visualizes how **borrow availability** correlates with **price movement**.

It offers two complementary visual modes:

| Mode | Description |
|------|--------------|
| 🟩 **Overview Grid** | Multi-symbol dashboard with `Price`, `Borrow Size`, and cumulative `BZAP` |
| 🟦 **Detail Navigator** | Interactive symbol-by-symbol exploration with keyboard shortcuts |

The launcher (`watchlist_launcher_gui.py`) provides a complete GUI interface — no command line needed.

---

## ✨ Key Features

- 🔍 **Automatic CSV Detection**  
  Detects separators and header rows even in inconsistent files.

- 🧩 **Smart Column Recognition**  
  Finds the correct columns for *Symbol*, *Price* (Mark/Last/Close), and *Approx Borrow Size* automatically.

- ⏱ **Two X-Axis Modes**  
  - `time`: true timeline (with spacing)  
  - `index`: equal spacing  

- ⚙️ **Dynamic Borrow Controls**  
  Borrow floor (`none`, `10th percentile`, `min`) and zoom factor for highlighting critical areas.

- 💹 **BZAP – Borrow Size Weighted Average Price**  
  - *BZAP (overall)* – global average  
  - *BZAP (cum.)* – cumulative average  
  - *BZAP (10-roll)* – rolling weighted means, it takes only the last 10 data pionts into acount

- 🎛 **Interactive Detail Navigation**
  | Key | Action |
  |-----|--------|
  | `A / D` or `← / →` | Previous / next symbol |
  | `Z` | Toggle Δ/absolute borrow |
  | `+ / -` | Zoom in/out bottom range |
  | `X` | Cycle floor mode (none → p10 → min) |
  | `Q` | Quit detail view |

- 💾 **Preset System**  
  - Saves configuration automatically to `user_presets.json`  
  - Manual save/load from the GUI  
  - Persists between sessions

- ⏰ **Required data format**

  - 2023-10-24 or 2023_10_24 or 20231024 or 24-10-2023 or 24.10.2023
  - _08-30. or -08-30. or _0830. or -0830
  - .csv

   excample: 24-10-2023-08-30.csv
    
---

## 🪟 GUI Launcher

Run the GUI to configure everything interactively:

```bash
python watchlist_launcher_gui.py
```

You can:

Browse for your analysis script and data folder

Adjust parameters (days, topN, ranking, modes, etc.)

Choose one of three run modes:

🟩 Run Overview only

🟦 Run Detail only

🟨 Run Both

The Command Preview updates live so you can see the exact CLI call.

---

🧮 The BZAP Formula

Borrow-Weighted Average Price (BZAP):

BZAP = ∑ (Price i ​× Borrow i​) / ∑ Borrow i​


---

| -----------------Overview Grid------------------ | -------------------Detail View-------------------- |

| <img width="321" height="258" alt="image" src="https://github.com/user-attachments/assets/c673190b-bf56-4d99-a90b-37e2e632f679" /> | <img width="335" height="258" alt="image" src="https://github.com/user-attachments/assets/5fd6093c-aa5a-4018-af7a-4d071cae7d82" /> |

