FROM continuumio/miniconda3

WORKDIR /app

# 1) Copy ONLY the env file first
COPY environment-top.yml .

# 2) Create the conda env
RUN conda env create -f environment-top.yml && \
    conda clean -a -y

# 3) Copy the rest of the code
COPY . .

# 4) Install the package
RUN conda run -n masters pip install -e .

# 5) Set pyhton path to src
ENV PYTHONPATH=/app/src

# 6) Default command (can be overridden at runtime)
CMD ["python", "scripts/evaluation/run_evaluation_pipeline.py"]