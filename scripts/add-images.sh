#!/bin/bash
set -e

if [ $# -ne 1 ];
then
    echo "usage: $0 <image-folder>"
    exit 1
fi

# Check to see if the 'leitmotiv_data' volume exists.  Create it if it doesn't.
volumes=$(docker volume ls -q)
found=0
for v in $volumes;
do
    if [ $v == "leitmotiv_data" ];
    then
        found=1
    fi
done

if [ $found -eq 0 ];
then
    _=$(docker volume create leitmotiv_data)
fi

echo "Adding images in $1"

# Run leitmotiv's "add-directory" subcommand, making sure that the directory is
# accessible to the container as a read-only directory.  It is immediately
# followed by a "rebuild-index" subcommand.
docker run --rm \
           --volume leitmotiv_data:/home/app/library \
           --volume "$1":/images:ro \
           leitmotiv add-directory /images

docker run --rm \
           --volume leitmotiv_data:/home/app/library \
           leitmotiv build-index
