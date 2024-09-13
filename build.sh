#!/bin/bash
pip install --upgrade pip
pip install setuptools wheel
apt-get update && apt-get install -y python3-distutils
pip install -r requirements.txt
