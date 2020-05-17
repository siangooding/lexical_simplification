FROM tensorflow/tensorflow:latest-gpu


COPY ./requirements.txt /requirements.txt
COPY ./python-download.py /python-download.py
COPY ./textrank /textrank
COPY ./setup.py /setup.py

RUN pip install -r requirements.txt
RUN python3 -m spacy download en
RUN python3 -m spacy download en_core_web_md
RUN python3 python-download.py
RUN cd /textrank && python setup.py install

RUN apt-get install -y git
RUN git clone https://github.com/nishkalavallabhi/OneStopEnglishCorpus.git
RUN mkdir onestop
RUN mkdir /onestop/adv-text-all
RUN mv /OneStopEnglishCorpus/Texts-SeparatedByReadingLevel/Adv-Txt/*.txt /onestop/adv-text-all


COPY . /
RUN python3 setup.py install

CMD cd /final_exp_simplify && python3 run_1_store_texts.py -t cascade