#!/bin/bash
docker run --rm -ti -v "$(pwd)":/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:current ". sourceme.sh; /bin/bash -i"