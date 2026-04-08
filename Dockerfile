# Dockerfile

# Use the official AWS Lambda Python 3.12 image
FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt first (for caching)
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
# Assuming your code is in an 'app' folder
COPY app/ ${LAMBDA_TASK_ROOT}/app/

# Set the CMD to your handler (file_name.function_name)
# Since your FastAPI app is wrapped in Mangum
CMD [ "app.main.handler" ]