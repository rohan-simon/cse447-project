FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
# RUN pip install tqdm
COPY requirements.txt /job/ 
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt