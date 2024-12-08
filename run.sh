#!/bin/bash
# Make sure this script has execute permissions: chmod +x run.sh
uvicorn app.main:app --host 0.0.0.0 --port 8000