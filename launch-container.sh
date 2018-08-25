#!/bin/bash
docker run --rm -ti -v "$(pwd)":/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:2017.2.0.r4 ". sourceme.sh; /bin/bash -i"
