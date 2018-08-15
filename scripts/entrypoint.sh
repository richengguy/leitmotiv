#!/bin/bash
# Simple entry point where if 'run' is called without arguments, it starts up
# the HTTP server.  Otherwise, it will pass the arguments along into the
# leitmotiv binary.
set -e

ENVBIN=${LEITMOTIV_ENV}/bin

if [ $# -eq 0 ];
then
    $ENVBIN/gunicorn -b 0.0.0.0:5000 leitmotiv.webui:app
else
    $ENVBIN/leitmotiv $@
fi
