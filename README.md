# Ki-67 in the T1D Pancreas

A one-page web app for the **UF AI Passport Impact Project**: Ki-67 protein density in endocrine
and exocrine pancreas across type 1 diabetes (T1D) stages, and digital staining of islet and
Ki-67 markers on H&E-stained tissue sections.

**Quick start:** install [uv](https://docs.astral.sh/uv/), then double-click `run_app.bat` (Windows) or
`run_app.command` (Mac), or run `uv run streamlit run app.py` in this folder.

---

## The page

The project brief (problem, open question, donor cohort, data) is at the top, the approach and
success criteria are in the left panel, and the tools follow one step after the next. A timeline
fixed at the bottom shows which step you are on; click a step to jump to it.

| Step                     | What it does                                                                                                     |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **1 · Load sections**    | The immunostained section (brightfield IHC, or multiplex fluorescence) and its consecutive H&E section, from FUSION (default), a local path or an upload. Markers are read from the IHC file name; markers not listed are switched off |
| **2 · Register**         | Aligns the immunostain to the H&E over the whole section (rotation and flips included), with manual rotate / shift / scale if needed. **Go to the region with the most islets** picks a region to check |
| **3 · View a region**    | H&E and immunostain side by side (linked zoom) or overlaid; each region is aligned again (searches ~300 µm around the coarse fit) |
| **4 · Quantify Ki-67**   | Islets (INS/GCG), exocrine tissue, Ki-67+ nuclei per mm² (and Ki-67 index where nuclei are stained). Whole-section results build a cohort table with statistics against the donor spreadsheet. **Measure all donors** runs the whole FUSION project folder in one go |
| **5 · Digital staining** | The IHC (ground truth) highlighted on the registered H&E: islets cyan, glucagon lime, Ki-67+ nuclei yellow (or the IHC's own colours). Optional tools export tile pairs for training a model later |

### nPOD IHC stains (from the file name)

| Slide name contains | Insulin | Glucagon | Ki-67 | Nuclei |
| ------------------- | ------- | -------- | ----- | ------ |
| `INS_Ki67` (older)  | red     | —        | brown (DAB) | blue hematoxylin counterstain: nuclei counted, Ki-67 index reported |
| `INS_GCG_Ki67` (newer, e.g. 6593) | red | blue | brown (DAB) | **not stained** (faint purple in nuclei is glucagon background): no nuclei count, no Ki-67 index |

Tissue is outlined from the total of all stains and the sections are registered on it, so neither
depends on a nuclear stain. Compare donors on **Ki-67+ per mm²**, which every slide has.

Results carry a `method` column; **Measure all donors** re-measures rows from an older method.
`data/results/ki67_sections_v1_before_marker_fix.csv` is the first run (before markers were read per
slide): do not use it for INS_GCG_Ki67 slides.

---

## Run it on your computer

You need [uv](https://docs.astral.sh/uv/) once:

- **Windows** (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **Mac** (Terminal): `curl -LsSf https://astral.sh/uv/install.sh | sh`

Then, in a terminal inside this folder:

```bash
uv sync                        # installs the exact package versions from uv.lock
uv run streamlit run app.py    # opens http://localhost:8501 in your browser
```

Stop the app with **Ctrl + C** in the terminal.

---

## Data, privacy and credentials

- Slides are read in place; only the parts needed are loaded. Uploads are copied to `data/uploads/`.
- Results stay on your computer in `data/` (never uploaded to GitHub):
  `data/results/ki67_sections.csv` (cohort table) and `data/digital_staining/` (tile pairs).
- The app connects to FUSION (`https://fusionpub.rc.ufl.edu/api/v1`) by default; Athena also works.
  For private slides, sign in with your username and password or an **API key** (archive → your name →
  My account → API keys). Credentials are never saved to disk.
- After connecting, paste the **project folder** address once (e.g. `https://fusionpub.rc.ufl.edu/#user/…/folder/…`,
  a folder holding ND, sAAB, mAAB with H&E and IHC folders inside). Each section then offers a Folder and a Slide
  dropdown, and **Measure all donors** (step 4) can measure every donor straight from FUSION. A folder in someone
  else's user space must be shared with you first.
- Put the donor spreadsheet (`.xlsx`, one row per donor, `case_id` column) in `data/` or upload it under the
  cohort chart to compare Ki-67 across stages, sex or autoantibodies and correlate it with age, BMI, HbA1c or
  C-peptide.
- Do not upload images with patient identifiers anywhere.

---

## Project layout

```
app.py                              the one-page web app
src/impactproject/slides.py         whole-slide readers (local files and archive items)
src/impactproject/registration.py   fluorescence-to-H&E registration
src/impactproject/quantify.py       islet / exocrine segmentation and Ki-67 counts
src/impactproject/cohort.py         donor discovery, whole-section batch, donor-spreadsheet statistics
src/impactproject/digital_stain.py  painting the IHC onto H&E; training-pair export and scoring
src/impactproject/athena.py         Athena / FUSION connection
.streamlit/config.toml              theme and upload size
pyproject.toml, uv.lock             package list and exact versions (used by uv)
```
