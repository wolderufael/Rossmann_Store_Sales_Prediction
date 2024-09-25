# # Use the official Python image
# FROM python:3.9-buster

# # Set the working directory in the container
# WORKDIR /opt/ml_in_app

# # Copy the entire project directory into the working directory
# COPY . .

# # Install required packages
# RUN pip install -r requirements_prod.txt

# # Set the command to run your application
# CMD ["python", "api/api.py"]
FROM python:3.9-buster

# Add the application code
ADD . /opt/ml_in_app
WORKDIR /opt/ml_in_app

# Install Python dependencies
RUN pip install -r requirements_prod.txt

# Install gdown for downloading from Google Drive
RUN pip install gdown

# Download the model file using gdown 
RUN gdown https://drive.google.com/uc?id=12oxV8ajG6FnuCDiogPYPHKdI84-1VUmm -O models/random_forrest_model-25-09-2024-05-03-20-00.pkl

# Command to run the application
CMD ["python", "api/api.py"]

# https://drive.google.com/file/d/12oxV8ajG6FnuCDiogPYPHKdI84-1VUmm/view?usp=sharing