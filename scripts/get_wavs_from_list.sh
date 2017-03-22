#!/bin/bash

function usage {
  echo "usage: $@ <list-file> <corpus-dir>"
  echo "  list-file      file containing list of files of interest"
  echo "  corpus-dir     directory where files of interest can be found"
  echo "  out-dir        output directory"
}

if [ $# -ne 3 ]; then
  usage;
  exit 1;
fi

echo "Copying files from $2 to $3"
cat $1 | while read FILENAME;
do find $2 -name "*$FILENAME" -type f -exec cp '{}' $3 \; ; done;
