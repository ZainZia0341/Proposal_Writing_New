# Dockerfile

# Use the official AWS Lambda Python 3.12 image
FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt first (for caching)
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application package and the local Uvicorn helper.
# Lambda uses app.main.handler.
# The repo-root main.py is only for local development convenience.
COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY main.py ${LAMBDA_TASK_ROOT}/main.py

# Lambda entrypoint exposed via Mangum in app.main
CMD [ "app.main.handler" ]
