# # Step 1: Use a base image with Python 3.9
# FROM python:3.9-slim

# # # Step 2: Set the working directory inside the container
# WORKDIR /app

# # # Step 3: Copy the requirements.txt into the container
# COPY requirements.txt .

# # # Step 4: Install the dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # # Step 5: Copy the rest of your application code to the container
# COPY . .

# # # Step 6: Expose the port on which Flask will run
# EXPOSE 5000

# # # Step 7: Set the command to run the API
# CMD ["python", "api/api.py"]
FROM python:3.9-buster

# Copy the entire project directory into /opt/ml_in_app in the container
ADD . /opt/ml_in_app

# Set the working directory
WORKDIR /opt/ml_in_app/api

# Install required packages
RUN pip install -r ../requirements_prod.txt

# Command to run the application
CMD ["python", "api.py"]
