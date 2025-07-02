# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Dash runs on
EXPOSE 8080

# Run the Dash app
CMD ["python", "app.py"]
