#!/bin/bash

current_dir=$(pwd)
script_dir=$(dirname $0)
cd $script_dir

FILE=net1.p.gz
URL=https://www.dropbox.com/s/nzhurlfha1b6pww/net1.p.gz?dl=0
CHECKSUM=00388c72b3a8513e4fb35b0096b915b00cae4c4ca74d82d161c74774c92c625d5305681e902675804ce4473901816d89029b8e27e6120b2aff3c4d7079e42274

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
