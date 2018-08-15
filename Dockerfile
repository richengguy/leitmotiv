# NOTE: This setup is based off this article:
# https://beenje.github.io/blog/posts/docker-and-conda/

FROM continuumio/miniconda3:4.5.4 as base

# Create a dedicated user so that leitmotiv doesn't run as root in the
# container.
RUN adduser --gecos "Leitmotiv User" --disabled-password app

# Switch over to the leitmotiv user and update the path accordingly.
USER app
WORKDIR /home/app

# Create the conda environment.
COPY --chown=app:app environment.yml /home/app/environment.yml
RUN conda env create

# Now, pull in the leitmotiv source and install it.
COPY ./leitmotiv /home/app/leitmotiv/leitmotiv
COPY ./testing /home/app/leitmotiv/testing
COPY ./setup.py /home/app/leitmotiv/setup.py

USER root
RUN chown -R app:app /home/app/leitmotiv
USER app

ENV LEITMOTIV_ENV /home/app/.conda/envs/leitmotiv
RUN ${LEITMOTIV_ENV}/bin/pip install --upgrade pip setuptools && \
    ${LEITMOTIV_ENV}/bin/pip install gunicorn && \
    ${LEITMOTIV_ENV}/bin/pip install -e ./leitmotiv/.

# Create a temporary test container to ensure that the tests all pass.
FROM base as test-env
RUN ${LEITMOTIV_ENV}/bin/pip install pytest
RUN ${LEITMOTIV_ENV}/bin/pytest ./leitmotiv/testing/cholesky.py \
                                ./leitmotiv/testing/io_utils.py \
                                ./leitmotiv/testing/linalg.py

# Create the final leitmotiv container.
FROM base as leitmotiv
COPY --chown=app:app config.yml .
COPY --chown=app:app ./scripts/entrypoint.sh .
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN mkdir /home/app/library && chown app:app library

ENTRYPOINT [ "/usr/bin/tini", "/home/app/entrypoint.sh", "--" ]
