FROM kaggle/python
WORKDIR /molecule
COPY . /molecule/.
RUN python setup.py develop

RUN bash