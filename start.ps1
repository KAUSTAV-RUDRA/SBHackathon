# Startup script for Internal Policy Agent
Write-Host "🚀 Starting Internal Policy Agent..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "   Please run: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# Activate virtual environment
& .\venv\Scripts\Activate.ps1

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  .env file not found. Creating template..." -ForegroundColor Yellow
    "GEMINI_API_KEY=your_api_key_here`nAPI_URL=http://localhost:8000" | Out-File -FilePath .env -Encoding utf8
    Write-Host "   Please edit .env and add your GEMINI_API_KEY" -ForegroundColor Yellow
    Write-Host ""
}

# Start FastAPI backend
Write-Host "📡 Starting FastAPI Backend (http://localhost:8000)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\venv\Scripts\Activate.ps1; Write-Host 'FastAPI Backend - http://localhost:8000' -ForegroundColor Green; python main.py"

# Wait a bit for backend to start
Start-Sleep -Seconds 3

# Start Streamlit frontend
Write-Host "🎨 Starting Streamlit Frontend (http://localhost:8501)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\venv\Scripts\Activate.ps1; Write-Host 'Streamlit Frontend - http://localhost:8501' -ForegroundColor Green; streamlit run app.py"

Write-Host ""
Write-Host "✅ Servers are starting!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Access the application:" -ForegroundColor Cyan
Write-Host "   Frontend: http://localhost:8501" -ForegroundColor White
Write-Host "   API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  Note: Make sure to set your GEMINI_API_KEY in .env file" -ForegroundColor Yellow
Write-Host ""

