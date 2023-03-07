#! /bin/bash

if [ $1 == 'export' ]; then
    conda env export --no-builds | grep -v "prefix" > environment.yaml
elif [ $1 == 'create' ]; then
    conda env create -f environment.yaml
elif [ $1 == 'update' ]; then
    conda env update --file environment.yaml --prune
else
    echo "support 'create', 'export' or 'update' only!"
fi
