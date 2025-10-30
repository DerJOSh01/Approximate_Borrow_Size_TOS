# How to run the Programm on Codespace

On the github page, click on the `<>Code` button and select your Codespace server in **main** branch (**blue**).  
If not existing, create a new Server (**green**).

<br>

![alt text](/pictures/image-2.png)

<br>

In the new opened window, wait for arround 1-2 minutes (First startup only), so that the server can setup automatically.  
After that, in Terminal at the bottom, type 
```bash
make app-up
```
and hit Enter. Wait for the server to load the programm. If any popup´s appear, allow them.  
In the bottom right of your screen, a popup should apper with a button `open in Browser...` - click on it to open the programm in a new tab.  
If you´ve misses the popup, goto `Ports` and open it manually.  

<br>

![alt text](/pictures/image-1.png) 

<br>

---
# 📄 Compatible files and content

The files have to be in **.csv** format *and* correct name syntax (see Key Features).

Necessary columns for evaluation (**only one** each type):
| Type | Column Name |
|------|-------------|
| Price | **last, mark**, price, close |
| Stock | **symbol**, ticker, name |
| Borrow Size | **approx borrow size** |

Additional columns and their order does´nt matter.

<br>

---

# 💾 File upload

To upload files for evaluation, go to your Codespace. Drag and drop the files into `/data` folder.  

To save the files and changes you made, click on the shown symbol and then **✓ Commit**!  

![alt text](/pictures/image.png)  

After that, a window on the right side will open where you have to write a commit message. Once done, you can close the window and the files are saved.

**All existing files in `/data` can be savely deleted - they are for demonstration only.**

<br>

---

# 🛑 Setting your Forked tradenote-selfhost Repository to Private
Forked repositories on GitHub are public by default, which can pose a security risk if you commit sensitive files. To protect your data, convert your forked repository to private by making it standalone first. Follow these steps:

- **Navigate to Repository Settings**: Go to your forked repository on GitHub and click on the **Settings** tab at the top.
- **Leave the Fork Network**: Scroll down to the **Danger Zone** section at the bottom of the General settings page. Click the **Leave fork network** button to unlink your repository from the original fork network, transforming it into a standalone repository. Confirm the action when prompted.
- **Change Visibility**: Once unlinked, the **Change visibility** button in the Danger Zone will become available. Click it, select **Change to private**, and confirm the change. Your repository is now private and inaccessible to the public.

### ⟳ Updating Your Private Repository with latest changes from this repository
After converting your forked repository to private, you may still want to incorporate updates from the original repository. To do this, use the make command:

```bash
make sync-code
```
- Preserves Your Files: Any additional files or folders you created will remain untouched.
- Overwrites Matching Files: If files exist both in your repository and the upstream source, they will be overwritten with the latest version.
- Best Practice: Before committing changes to your private repository, run a git diff to review modifications and ensure nothing important is unintentionally overwritten.
---

# 📈 Approx Borrow Size Analysis

A powerful and intuitive **visual analytics suite** for exploring  
the relationship between **Price** and **Approx Borrow Size** in Thinkorswim.

This project combines:
- A robust Python backend for data parsing and visualization, and  
- A modern Tkinter GUI launcher for quick configuration and presets.

---

## 🧭 Overview

**Approx Borrow Size Analysis** loads multiple CSV snapshots exported from your trading watchlist *or* scanner in Thinkorswim, merges them by timestamp, and visualizes how **share availability** correlates with **price movement**.

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
  - Displayes error messages in Output section, if anything can´t be processed.

- ⏱ **Two X-Axis Modes**  
  - `time`: true timeline (with spacing)  
  - `index`: equal spacing  

- ⚙️ **Dynamic Borrow Controls**  
  Borrow floor (`none`, `10th percentile`, `min`) and zoom factor to resize bar chart.

- 💹 **BZAP – Borrow Size Weighted Average Price**  
  - *BZAP (overall)* – global average  
  - *BZAP (cum.)* – cumulative average  
  - *BZAP (10-roll)* – rolling weighted means, it takes only the last 10 data pionts into acount

- 🎛 **Interactive Detail Navigation**
  | Key | Action |
  |-----|--------|
  | `A / D` or `← / →` | Previous / next symbol |
  | `Z` | Toggle Δ (Delta) / absolute borrow |
  | `+ / -` | Zoom in/out bottom range |
  | `X` | Cycle floor mode (none → p10 → min) |
  | `Q` | Quit detail view |

- 💾 **Preset System**  
  - Saves configuration automatically to `user_presets.json`  
  - Manual save from the GUI  
  - Persists between sessions

- ⏰ **Required data format**

  - 2023-10-24 or 2023_10_24 or 20231024 or 24-10-2023 or 24.10.2023
  - _08-30. or -08-30. or _0830. or -0830
  - .csv

    excample: `2023-10-24-08-30.csv` , `2023-10-24_0830.csv`
    
---

## 🪟 GUI Launcher

Run the GUI to configure everything interactively:

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

| ---------------------Overview Grid--------------------- | ------------------------Detail View----------------------- |

| <img width="321" height="258" alt="image" src="https://github.com/user-attachments/assets/c673190b-bf56-4d99-a90b-37e2e632f679" /> | <img width="335" height="258" alt="image" src="https://github.com/user-attachments/assets/5fd6093c-aa5a-4018-af7a-4d071cae7d82" /> |
