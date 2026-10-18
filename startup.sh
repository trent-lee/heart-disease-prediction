#!/bin/bash
# Azure App Service startup command:  bash startup.sh
python -m streamlit run app.py --server.port "${PORT:-8000}" --server.address 0.0.0.0
