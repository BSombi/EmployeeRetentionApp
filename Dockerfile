# Dockerfile

# pull the official docker image
# Install minimal Python 3.
FROM python:3.10.6

EXPOSE 8501

WORKDIR /app

RUN pip install git+https://github.com/pycaret/pycaret.git#egg=pycaret

COPY . .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "employeeretention_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
