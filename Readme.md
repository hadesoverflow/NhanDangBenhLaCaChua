# Huong Dan Chay NhanDangBenhLaCaChua

Project hien tai chi can 2 folder chinh:

```text
backend/   FastAPI + ONNX Runtime
frontend/  ReactJS + Vite
```

Model se lay tu Hugging Face thong qua environment variables. Cac file model local `.onnx`, `.onnx.data`, `.pth`, `.pt` da duoc dua vao `.gitignore` de khong day len GitHub.

## 1. Chuan Bi Model URL

Can 5 URL model:

```ini
GATE_MODEL_URL=https://huggingface.co/nonametd/disease_leaf_tomato/resolve/main/resnet50_tomato_leaf_gate.onnx
GATE_MODEL_DATA_URL=https://huggingface.co/nonametd/disease_leaf_tomato/resolve/main/resnet50_tomato_leaf_gate.onnx.data
DETECT_MODEL_URL=https://huggingface.co/nonametd/disease_leaf_tomato/resolve/main/model_detect_plant_leaf.onnx
DISEASE_MODEL_URL=https://huggingface.co/nonametd/disease_leaf_tomato/resolve/main/resnet50_tomato.onnx
DISEASE_MODEL_DATA_URL=https://huggingface.co/nonametd/disease_leaf_tomato/resolve/main/resnet50_tomato.onnx.data
```

Luu y: dung `resolve/main`, khong dung `blob/main`.

## 2. Chay Backend Local Bang Venv

Mo PowerShell tai folder goc:

```powershell
cd E:\NhanDangBenhLaCaChua
```

Tao venv rieng cho backend:

```powershell
cd backend
python -m venv .venv
```

Kich hoat venv:

```powershell
.\.venv\Scripts\Activate.ps1
```

Neu PowerShell chan script, chay lenh nay mot lan:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Cai thu vien backend:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Tao file `.env` trong folder `backend/`:

```powershell
Copy-Item .env.example .env
```

Mo file `backend\.env` va sua 5 URL model neu can. Sau do chay backend:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8010
```

Kiem tra backend:

```text
http://localhost:8010/health
```

Neu model URL dung, ket qua se co:

```json
{
  "status": "ok"
}
```

Lan dau tien co the lau vi backend tai model tu Hugging Face ve cache.

## 3. Chay Frontend React Local

Mo PowerShell khac:

```powershell
cd E:\NhanDangBenhLaCaChua\frontend
```

Cai package:

```powershell
npm install
```

Tao file `.env` cho frontend:

```powershell
Copy-Item .env.example .env
```

Noi dung mac dinh:

```ini
VITE_BACKEND_URL=http://localhost:8010
```

Chay frontend:

```powershell
npm run dev
```

Mo trinh duyet:

```text
http://localhost:5173
```

Upload anh va bam `Predict`.

## 4. Test API Bang Command Line

Khi backend dang chay:

```powershell
cd E:\NhanDangBenhLaCaChua
python predict_tomato_pipeline.py --image path\to\image.jpg --backend-url http://localhost:8010
```

Neu muon xem probability:

```powershell
python predict_tomato_pipeline.py --image path\to\image.jpg --backend-url http://localhost:8010 --include-probabilities
```

## 5. Deploy Len Render

Project da co:

```text
render.yaml
```

Len Render:

```text
New + -> Blueprint
```

Chon GitHub repo. Render se tao:

```text
tomato-onnx-backend
tomato-onnx-frontend
```

Trong service `tomato-onnx-backend`, dien 5 env vars:

```text
GATE_MODEL_URL
GATE_MODEL_DATA_URL
DETECT_MODEL_URL
DISEASE_MODEL_URL
DISEASE_MODEL_DATA_URL
```

Frontend se tu lay `VITE_BACKEND_URL` tu backend service theo `render.yaml`.

## 6. Cac Thu Khong Can Commit

Khong commit cac thu sau:

```text
backend/.venv/
frontend/node_modules/
frontend/dist/
*.onnx
*.onnx.data
*.pth
*.pt
__pycache__/
```

Neu da lo add model vao git, go khoi index nhung giu file local:

```powershell
git rm --cached -r model_classify_leaf_domain model_classify_disease_tomato model_detect_leaf_tomato
```

Sau do commit lai.
