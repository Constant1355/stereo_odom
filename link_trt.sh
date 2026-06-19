#!/bin/bash
cd ~/Documents/stereo_odom
source stereo_odom/bin/activate
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
ln -sf /usr/lib/python3.12/dist-packages/tensorrt "$SITE/tensorrt"
echo "Linked tensorrt to $SITE/tensorrt"
python3 -c "import tensorrt; print('TRT:', tensorrt.__version__)"
