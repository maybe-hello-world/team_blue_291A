# standalone
uvicorn main:app --reload --host 0.0.0.0 --proxy-headers --port 80

# behind reverse proxy
# uvicorn main:app --reload --host 127.0.0.1 --proxy-headers --port 8081