# Use the official Python image
FROM python:3.9-buster

# Set the working directory in the container
WORKDIR /opt/ml_in_app

# Copy the entire project directory into the working directory
COPY . .

# Install required packages
RUN pip install -r requirements_prod.txt

# Set the command to run your application
CMD ["python", "api/api.py"]
