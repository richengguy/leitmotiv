#!/bin/bash
set -e

function find_entry() {
    found=0
    for v in $2;
    do
        if [ $v == "$1" ];
        then
            found=1
            break
        fi
    done
    echo $found
}

if [ $# -eq 0 ];
then
    echo "Usage: $0 [ --start | --stop ]"
    echo "  --start - start the server container if it isn't running"
    echo "  --stop  - stop the server container if it's running"
    exit 1
fi

# Check to see if the 'leitmotiv_data' volume exists.  The application needs to
# be initialized if it doesn't.
volumes=$(docker volume ls -q)
containers=$(docker ps --format "{{ .Names }}")

has_volume=$(find_entry "leitmotiv_data" $volumes)
is_running=$(find_entry "leitmotiv" $containers)

if [ $has_volume -eq 0 ];
then
    echo "Please initialize the leitmotiv library using 'add-images.sh'."
    exit 1
fi

# Run the container as a detached process.
case $1 in
    "--start")
        if [ $is_running -eq 1 ];
        then
            echo "Container is already running."
            exit 0
        else
            echo "Starting container."
        fi
        docker run --rm \
                   --detach \
                   --publish 5000:5000 \
                   --volume leitmotiv_data:/home/app/library \
                   --name leitmotiv \
                   leitmotiv
        ;;
    "--stop")
        if [ $is_running -eq 0 ];
        then
            echo "Container is not running."
            exit 0
        else
            echo "Stopping container."
        fi
        docker container stop leitmotiv
        ;;
esac
