# Heart Disease Prediction System — Setup & Deployment Guide

## 1. Overview

In this project you take a machine-learning model from your own machine to a live website, the same way a professional ML engineer would:

1. **Clone** the project from GitHub onto your computer or virtual machine (VM).
2. **Run** the Streamlit dashboard locally.
3. **Train** three models (XGBoost, Random Forest, Decision Tree) and **track** every experiment with **MLflow**.
4. **Push** your work to your own GitHub repository.
5. **Deploy** automatically to a free **Azure App Service** using a **GitHub Actions CI/CD pipeline**: every push to `main` tests the model and redeploys the site.

```
 your laptop / VM                    GitHub                         Azure (free tier)
┌─────────────────────┐  git push  ┌──────────────────────┐ deploy ┌──────────────────────┐
│ python main.py      │ ─────────► │ your repository      │ ─────► │ App Service (Linux)  │
│  └─ MLflow tracking │            │ GitHub Actions:      │        │ streamlit run app.py │
│ streamlit run app.py│            │  test model → deploy │        │ https://<app>.azure… │
└─────────────────────┘            └──────────────────────┘        └──────────────────────┘
```

| Part | Technology |
|---|---|
| Language | Python 3.12 |
| Machine learning | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Experiment tracking | MLflow (local SQLite database) |
| Dashboard | Streamlit + Altair charts |
| Source control | Git + GitHub |
| CI/CD | GitHub Actions |
| Hosting | Azure App Service, Linux, Free F1 plan |

**Cost:** everything in this guide uses free tiers. **Time:** about 60–90 minutes the first time.

## 2. Prerequisites

**Accounts**

- A **GitHub** account: https://github.com/signup
- An **Azure free account**: https://azure.microsoft.com/free. A card is needed for identity verification, but the Free F1 App Service plan used here costs nothing.

**Software on your machine or VM**

| Tool | Check it is installed |
|---|---|
| Git | `git --version` |
| Python **3.12** | `python3.12 --version` (Windows: `py -3.12 --version`) |
| A code editor (VS Code recommended) | optional |
| GitHub CLI (`gh`) | optional, makes logging in to GitHub easier |

**Ubuntu / Debian VM (24.04 ships Python 3.12):**

```bash
sudo apt update
sudo apt install -y git python3.12 python3.12-venv
```

On Ubuntu 22.04, add Python 3.12 first: `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update`.

**macOS (with Homebrew):**

```bash
brew install git python@3.12 gh
```

**Windows:** install Git from https://git-scm.com and Python 3.12 from https://www.python.org/downloads (tick **"Add python.exe to PATH"**).

> Use Python 3.12 both locally and on Azure. The trained model is saved with specific library versions, and matching Python versions avoids loading problems.

## 3. Get the code

Starter repository: **https://github.com/Abhi1727/heart-disease-azure-starter**

**1. Download the code.** Open a terminal in the folder where you keep your projects (for example `cd ~/Projects`, or `cd Documents` on Windows) and paste:

```bash
git clone https://github.com/Abhi1727/heart-disease-azure-starter.git
cd heart-disease-azure-starter
```

That is all you need to run and train the model locally (steps 5–8). The same commands work on macOS, Linux and Windows PowerShell.

**2. Make it your own repository.** To push your changes and deploy to *your* Azure account you need a copy on *your* GitHub account. Choose one option:

**Option A — From the terminal (recommended, needs the GitHub CLI `gh`)**

Run these inside the `heart-disease-azure-starter` folder. Nothing needs editing: `gh` uses the account you log in with.

```bash
gh auth login --scopes workflow
gh auth setup-git
git remote rename origin upstream
gh repo create heart-disease-azure-starter --public --source . --remote origin --push
```

- `gh auth login` asks a few questions: choose **GitHub.com → HTTPS → Login with a web browser**, and enter the code shown.
- `git remote rename origin upstream` keeps a link to the original starter, so you can fetch future updates with `git pull upstream main`.
- `gh repo create` creates `https://github.com/<your-github-username>/heart-disease-azure-starter` and pushes the code to it. Use `--private` instead of `--public` if you prefer.

**Option B — From the website ("Use this template")**

1. Open https://github.com/Abhi1727/heart-disease-azure-starter and click **Use this template → Create a new repository**.
2. Keep the name **`heart-disease-azure-starter`**, choose **Public** or **Private**, and click **Create repository**.
3. Replace the folder from step 1 with a clone of **your** copy. Change `<your-github-username>` to your GitHub username before pressing Enter:

```bash
cd ..
rm -rf heart-disease-azure-starter      # Windows PowerShell: Remove-Item -Recurse -Force heart-disease-azure-starter
git clone https://github.com/<your-github-username>/heart-disease-azure-starter.git
cd heart-disease-azure-starter
```

**Check it worked:**

```bash
git remote -v
```

`origin` must point to **your** GitHub account. If it still shows `Abhi1727`, you are pushing to the original starter, which will be refused.

From now on, **all commands run inside the `heart-disease-azure-starter` folder**.

## 4. Project structure

```
heart-disease-azure-starter/
├── .github/workflows/
│   └── deploy.yml             ← CI/CD: test the model, then deploy to Azure
├── app.py                     ← Streamlit entry point (navigation, theme)
├── dashboard/                 ← dashboard pages: predict, performance, data, guide
├── assets/                    ← logo
├── docs/                      ← this guide (Markdown + Word)
├── Data/heart.csv             ← training dataset (918 patients)
├── config.py                  ← paths, column names, MLflow settings
├── data_loader.py             ← pipeline step 1: load the CSV
├── preprocessing.py           ← pipeline step 2: clean, split, scale + one-hot encode
├── train_models.py            ← pipeline step 3: tune XGBoost / Random Forest / Decision Tree
├── evaluate_models.py         ← pipeline step 4: score, log to MLflow, save the best model
├── main.py                    ← runs steps 1–4
├── models/                    ← best_model.pkl + metrics.json (what the dashboard serves)
├── requirements.txt           ← packages the dashboard needs (installed on Azure)
├── requirements-dev.txt       ← + training and MLflow packages (local only)
├── startup.sh                 ← the command Azure uses to start Streamlit
└── .streamlit/config.toml     ← theme and server settings
```

**How the pieces fit together:** `main.py` trains models and writes `models/best_model.pkl` and `models/metrics.json`. The dashboard (`app.py`) only *reads* those two files. The saved model is a complete scikit-learn **pipeline** (scaling → encoding → feature selection → model), so the dashboard can pass raw values such as `Sex = "M"` and get exactly the same preprocessing used in training.

## 5. Run the dashboard locally

**1. Create and activate a virtual environment** (an isolated set of Python packages for this project):

```bash
# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt now starts with `(.venv)`. Run `source .venv/bin/activate` (or the Windows equivalent) again whenever you open a new terminal.

**2. Install the packages** (dashboard + training + MLflow):

```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
```

**3. Start Streamlit:**

```bash
streamlit run app.py
```

Open **http://localhost:8501**. The repository already contains a trained model, so the dashboard works immediately. Stop it with **Ctrl + C**.

**Running on a cloud VM?** Either:

- tunnel the port from your laptop: `ssh -L 8501:localhost:8501 <user>@<vm-ip>`, then open http://localhost:8501 on your laptop, **or**
- start with `streamlit run app.py --server.address 0.0.0.0`, allow inbound TCP port **8501** in the VM's firewall / network security group, and open `http://<vm-ip>:8501`.

## 6. Train the model

```bash
python main.py
```

What happens:

1. `data_loader.py` reads `Data/heart.csv`.
2. `preprocessing.py` removes missing values and duplicates and makes a stratified 80/20 train/test split.
3. `train_models.py` builds a pipeline for each algorithm — **scale numbers → one-hot encode categories → select the most important features → balance classes with SMOTE → model** — and tunes it with 5-fold cross-validated grid search.
4. `evaluate_models.py` scores every model on the test set, logs everything to MLflow, saves the best one to `models/best_model.pkl`, writes `models/metrics.json`, and registers the model in the MLflow Model Registry.

Output you should see at the end:

```
🏆 Best Model: XGBoost (accuracy = 0.8696)
📚 Registered in MLflow as 'heart-disease-classifier' version 1
✅ Pipeline Complete!
```

Files created or updated:

| File / folder | What it is | Commit it? |
|---|---|---|
| `models/best_model.pkl` | the model the dashboard serves | ✅ yes |
| `models/metrics.json` | scores and chart data for the dashboard | ✅ yes |
| `model_graphs/*.png` | confusion matrix and ROC plots (also stored in MLflow) | ❌ no |
| `mlflow.db`, `mlruns/` | your local MLflow experiment history | ❌ no |

Restart Streamlit (**Ctrl + C**, then `streamlit run app.py`) to load the new model — the dashboard caches the model in memory.

## 7. Track experiments with MLflow

MLflow records **every training run**: the hyper-parameters used, the scores, the plots and the model files. This lets you compare experiments instead of guessing.

Open a **second terminal**, activate the environment, and start the MLflow UI:

```bash
cd heart-disease-azure-starter
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open **http://localhost:5001**.

> On macOS, port 5000 is used by AirPlay Receiver and returns "403 Forbidden", which is why this guide uses port **5001**.

What to explore:

- **Experiments → heart-disease-prediction**: one parent run (`training-pipeline`) per `python main.py`, with a child run per algorithm.
- Click a child run: **Parameters** (best hyper-parameters), **Metrics** (`cv_accuracy`, `test_accuracy`, `test_roc_auc`, …) and **Artifacts** (plots and the saved model).
- Tick several runs and click **Compare** to see scores side by side or as charts.
- **Models → heart-disease-classifier**: every training run registers its best model as a new **version** (v1, v2, …), giving you a history of what you could deploy.

## 8. Retrain and improve the model

A typical retraining loop:

1. **Change something** — one thing at a time, so you know what made the difference.
2. **Retrain** with a descriptive run name: `python main.py --run-name "rf-deeper-trees"`.
3. **Compare** the new run with older ones in MLflow.
4. **Check** the dashboard locally (restart Streamlit).
5. **Ship it** — commit `models/` and push (see steps 10 and 14). The pipeline redeploys automatically.

**Ideas to try**

*Change the hyper-parameter grid* in `train_models.py`:

```python
"RandomForest": {
    "model": RandomForestClassifier(random_state=RANDOM_STATE),
    "params": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [5, 8, 12],
        "model__min_samples_split": [2, 5],
    },
},
```

*Choose the served model by a different metric* — in a medical setting, missing a sick patient is costly, so **recall** may matter more than accuracy:

```bash
python main.py --metric recall --run-name "select-by-recall"
```

Valid metrics: `accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`.

*Add a new algorithm* — add an entry to `MODELS` in `train_models.py`:

```python
from sklearn.linear_model import LogisticRegression

"LogisticRegression": {
    "model": LogisticRegression(max_iter=1000),
    "params": {"model__C": [0.1, 1.0, 10.0]},
},
```

*Train on new data* — replace `Data/heart.csv` with a file that has the same columns.

> `RANDOM_STATE = 42` in `config.py` makes runs reproducible: the same code and data give the same model.

## 9. What to commit — and what never to commit

`.gitignore` files already handle this, but you should understand why.

| ✅ Commit | ❌ Never commit |
|---|---|
| Source code (`*.py`, `dashboard/`) | `.venv/` — your local Python environment (large, machine-specific) |
| `Data/heart.csv` (public dataset) | `mlflow.db`, `mlruns/` — local experiment history |
| `models/best_model.pkl`, `models/metrics.json` | `model_graphs/` — regenerated on every run |
| `requirements*.txt`, `startup.sh`, `.streamlit/` | `__pycache__/`, `.DS_Store` |
| `docs/`, `assets/`, `.github/workflows/` | **Secrets**: publish profiles (`*.PublishSettings`), `.env` files, passwords, API keys, tokens |

**Before every commit:**

```bash
git status                  # what changed?
git diff --stat             # how much changed?
```

If you see a file you do not recognise — especially anything containing passwords or keys — do **not** add it.

**Keep personal details out of the repository:**

- Azure credentials go into **GitHub Secrets**, never into files (step 13).
- To hide your email address in commits, use GitHub's private no-reply address (GitHub → Settings → Emails → *Keep my email addresses private*):

```bash
git config --global user.name  "Your Name"
git config --global user.email "<id>+<username>@users.noreply.github.com"
```

> If a secret is ever pushed by mistake, deleting the file is **not** enough — it stays in Git history. Reset it immediately (for a publish profile: App Service → Overview → **Reset publish profile**) and download a new one.

## 10. Push your changes to GitHub

**1. Log in to GitHub from the terminal (one time).** The easiest way is the GitHub CLI. The `workflow` scope is required because this repository contains a GitHub Actions workflow file:

```bash
gh auth login --scopes workflow
gh auth setup-git
```

(Without `gh`, use a Personal Access Token with **repo** and **workflow** scopes as your password when Git asks.)

**2. Commit and push:**

```bash
git status
git add .
git commit -m "Retrain model: select by recall"
git push origin main
```

**3. Verify** on github.com that your commit appears in your repository.

## 11. Create the Azure App Service (free tier)

1. Sign in to https://portal.azure.com.
2. **Create a resource group** (a folder for related Azure resources): search **Resource groups → + Create** → name it e.g. `rg-heart-app`, pick a region near you → **Review + create → Create**.
3. **Create the web app**: search **App Services → + Create → Web App**, then on the **Basics** tab:

| Field | Value |
|---|---|
| Subscription | your free subscription |
| Resource group | `rg-heart-app` |
| Name | a unique name, e.g. `heart-predictor-<yourname>` |
| Publish | **Code** |
| Runtime stack | **Python 3.12** (avoid versions marked *preview*) |
| Operating system | **Linux** |
| Region | same as the resource group |
| Pricing plan | create a new Linux plan → **Free F1** |

4. On the **Deployment** tab, leave **Continuous deployment disabled** — this repository already contains its own pipeline (step 13).
5. **Review + create → Create**, wait for *"Your deployment is complete"*, then **Go to resource**.
6. On **Overview**, copy the **Default domain**. This is your app's address: `https://<default-domain>`.

> If **Free F1** is not offered in your region, choose another region for both the plan and the app.

## 12. Configure the App Service

**1. Startup command and publishing credentials** — open **Settings → Configuration → General settings**:

- **Startup Command:** `bash startup.sh`
- **SCM Basic Auth Publishing Credentials:** **On** (needed to download the publish profile in the next step)
- Click **Save → Continue**.

`startup.sh` runs `python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0` — Streamlit listening on the port Azure expects.

**2. Build setting** — open **Settings → Environment variables → App settings → + Add**:

| Name | Value |
|---|---|
| `SCM_DO_BUILD_DURING_DEPLOYMENT` | `true` |

Click **Apply**, then **Apply → Confirm**. This tells Azure to run `pip install -r requirements.txt` every time new code arrives.

> Web sockets, which Streamlit needs, are always enabled on Linux App Services, so there is nothing to switch on. You will not see a Web sockets toggle; WebJobs can stay off.

## 13. Connect GitHub to Azure (CI/CD)

The workflow `.github/workflows/deploy.yml` runs on every push to `main`:

1. **Test** — installs `requirements.txt` on a fresh Linux machine and checks the saved model loads and predicts.
2. **Deploy** — uploads the committed files to your App Service, where Azure installs the packages and runs `startup.sh`.

It needs two pieces of information from you, stored safely in GitHub:

**1. Download the publish profile** — App Service → **Overview → Download publish profile**. This file is a password: never commit it, and delete it after the next step.

**2. Add it to GitHub** — in **your** repository: **Settings → Secrets and variables → Actions**:

- **Secrets** tab → **New repository secret**
  - Name: `AZURE_WEBAPP_PUBLISH_PROFILE`
  - Secret: open the downloaded file in a text editor and paste its **entire** contents
- **Variables** tab → **New repository variable**
  - Name: `AZURE_WEBAPP_NAME`
  - Value: your App Service name (e.g. `heart-predictor-<yourname>`)

**3. Check GitHub Actions is enabled** — open the **Actions** tab of your repository. If it shows a button such as **"I understand my workflows, go ahead and enable them"**, click it.

**4. Run the deployment** — **Actions → Deploy to Azure → Run workflow → Run workflow**. Wait for the green tick (about 3 minutes).

**5. Open your app** at `https://<default-domain>`. The very first start takes 1–3 minutes while Azure installs the packages.

From now on, **every `git push` to `main` redeploys automatically**.

> **Why not Azure's "Deployment Center → GitHub" button?** It generates a second workflow that duplicates this one without the model test, and it picks up whatever Python version the portal suggests. Use the provided workflow instead. If you already used Deployment Center, delete the workflow file it added (`.github/workflows/main_<app-name>.yml`) and follow the steps above.

**Advanced alternative — OpenID Connect instead of a publish profile.** If your organisation disables basic-auth publishing, create a Microsoft Entra app or user-assigned managed identity with a federated credential for your repository, give it the *Website Contributor* role on the App Service, and add the secrets `AZURE_CLIENT_ID`, `AZURE_TENANT_ID` and `AZURE_SUBSCRIPTION_ID` instead of `AZURE_WEBAPP_PUBLISH_PROFILE`. The workflow detects which method you configured.

## 14. Day-to-day workflow

Once everything is connected, improving the live model is a short loop:

```bash
cd heart-disease-azure-starter
source .venv/bin/activate

python main.py --run-name "describe-your-change"   # 1. retrain (tracked in MLflow)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # 2. compare runs
streamlit run app.py                               # 3. check the dashboard locally

git add .                                          # 4. ship it
git commit -m "Retrain: describe-your-change"
git push origin main                               # 5. GitHub Actions tests + deploys
```

Watch progress in the **Actions** tab of your repository, then refresh your Azure URL.

## 15. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `git push` rejected: *"refusing to allow an OAuth App to create or update workflow … without workflow scope"* | Your GitHub login lacks the `workflow` permission. Run `gh auth refresh -h github.com -s workflow` and `gh auth setup-git`, then push again. |
| Workflow finishes with *"Deployment skipped"* | The `AZURE_WEBAPP_NAME` variable or the `AZURE_WEBAPP_PUBLISH_PROFILE` secret is missing or misspelled (step 13). |
| **Download publish profile** is greyed out | Turn on **SCM Basic Auth Publishing Credentials** (step 12). |
| Deploy step fails with *401 Unauthorized* | The publish profile was reset or is outdated. Download it again and update the secret. |
| Site shows Azure's default *welcome / "waiting for your content"* page or *404* | The **Startup Command** is not `bash startup.sh`, or no deployment has run yet. |
| *503 Service Unavailable* or *Application Error* right after changing the Python version | The installed packages belong to the old version. Re-run the workflow (**Actions → Run workflow**). |
| Log shows `ModuleNotFoundError: No module named 'streamlit'` | `SCM_DO_BUILD_DURING_DEPLOYMENT=true` is missing (step 12). Add it and redeploy. |
| First page load takes a minute, or the app stops after heavy use | Free F1 apps sleep when idle and have 60 CPU-minutes per day. Wait, or upgrade to a Basic B1 plan. |
| Dashboard still shows the old model after retraining | Restart Streamlit locally; on Azure, make sure `models/` was committed and pushed. |
| `localhost:5000` shows *403 Forbidden* (macOS) | AirPlay Receiver uses port 5000 — run MLflow on port 5001. |
| `python3.12: command not found` | Install Python 3.12 (step 2). On Ubuntu 22.04 use the deadsnakes PPA. |

**Where to find logs on Azure:** App Service → **Log stream** (live output of `startup.sh`), or **Deployment Center → Logs** for each deployment.

## 16. Clean up

To stop everything and remove all Azure resources created in this guide, delete the resource group: **Resource groups → `rg-heart-app` → Delete resource group**, then type its name to confirm. This removes the App Service and its plan in one go.

You can also delete the GitHub secret and variable (**Settings → Secrets and variables → Actions**) if you no longer deploy from the repository.
