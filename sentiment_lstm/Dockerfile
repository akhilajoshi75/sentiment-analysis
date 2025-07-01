# Use slim Python base
FROM python:3.10-slim

# Set workdir inside container
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Start the sentiment API
CMD ["python", "sentiment_api.py"]

