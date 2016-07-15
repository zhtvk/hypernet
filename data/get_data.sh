#!/bin/bash

current_dir=$(pwd)
script_dir=$(dirname $0)
cd $script_dir

FILE=indian10p.p.gz
URL=https://www.dropbox.com/s/ueavju2mn3sthqe/indian10p.p.gz?dl=0
CHECKSUM=95674f9b2e90044ebd2a183e29fd73a8e782fd9a3a64eeeb559d206305ad7b0ca71947741320396f57b25e8b7d037426131cb6587c4ae69db7d1dda24a567116

if [ -f $FILE ]
then
  echo "File $FILE already exists. Checking SHA-512 sum..."
  checksum=`sha512sum $FILE | awk '{print $1}'`

  if [ "$checksum" = "$CHECKSUM" ]
  then
    echo "Checksum is correct."
    exit 0
  else
    echo "Checksum is incorrect."
  fi
fi

echo "Downloading file $FILE..."

wget $URL -O $FILE

echo "Done. Please run $0 again to verify checksum."

cd $current_dir
