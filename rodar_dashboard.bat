@echo off
cd /d %~dp0
if not exist saida\pedidos.parquet (
  echo Processando bases e treinando o modelo...
  python motor_churn.py
  if errorlevel 1 (
    echo.
    echo *** O processamento falhou. Veja a mensagem acima. O dashboard NAO foi aberto. ***
    pause
    exit /b 1
  )
)
streamlit run app_churn.py
