#!/bin/bash
docker run --rm -ti -v "$(pwd)":/home/fenics/shared -w /home/fenics/shared quay.io/fenicsproject/stable:2018.1.0.r3 ". sourceme.sh; /bin/bash -i"