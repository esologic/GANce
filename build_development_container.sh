#!/bin/bash

# Builds a standard `gance-develop` container for development.
# Locates the `:x` of the currently attached to display to be able to pass this to the container,
# so it can be written to

sudo docker-compose build --build-arg DISPLAY="$(w -hs | grep gdm | awk '{print $3}' | head -n 1)" gance-develop
sudo docker-compose up --detach gance-develop