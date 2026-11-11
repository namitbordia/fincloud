FROM public.ecr.aws/lambda/python:3.11

RUN dnf install -y libgomp && dnf clean all

COPY backend/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN python -m pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY backend ${LAMBDA_TASK_ROOT}/backend
COPY src ${LAMBDA_TASK_ROOT}/src

CMD ["backend.app.handler.lambda_handler"]
