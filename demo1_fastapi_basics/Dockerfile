# 1️⃣  Use a lightweight Python base image
FROM python:3.10-slim

# 2️⃣  Set working directory inside the container
WORKDIR /app

# 3️⃣  Copy & install dependencies FIRST (Docker layer caching!)
#     If requirements.txt hasn't changed, Docker skips this step
#     on subsequent builds – saving lots of time.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣  Copy application code
COPY . .

# 5️⃣  Expose the port the app listens on
EXPOSE 8000

# 6️⃣  Start the server
#     --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
