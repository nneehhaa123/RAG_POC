#!/bin/bash

notebook_directory="notebooks"
notebooks=$(ls -p $notebook_directory/*.ipynb)
for nb in $notebooks; do
    nbstripout < "$nb" > $notebook_directory/OUT.ipynb
    cmp -s "$nb" $notebook_directory/OUT.ipynb
    status=$?
    rm $notebook_directory/OUT.ipynb
    if [[ $status -ne 0 ]]; then
        echo "Notebook ""${nb}"" has not been stripped"
        exit 1
    fi
done
