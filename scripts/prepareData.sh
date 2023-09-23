#!/bin/bash
if [ ! -d "$1" ]; then
echo "$1 is not a directory."
return 1
fi

if [ -e "$2" ] && [ "$(ls -A $2)" ]; then
echo "$2 already exists and is not empty."
return 1
fi

flairPathFrom="$1"'/BraTS20_Training_???/BraTS20_Training_???_flair.nii.gz'
segPathFrom="$1"'/BraTS20_Training_???/BraTS20_Training_???_seg.nii.gz'
t1PathFrom="$1"'/BraTS20_Training_???/BraTS20_Training_???_t1.nii.gz'
t1cePathFrom="$1"'/BraTS20_Training_???/BraTS20_Training_???_t1ce.nii.gz'
t2PathFrom="$1"'/BraTS20_Training_???/BraTS20_Training_???_t2.nii.gz'

flairPathTo="$2"'/flair'
segPathTo="$2"'/seg'
t1PathTo="$2"'/t1'
t1cePathTo="$2"'/t1ce'
t2PathTo="$2"'/t2'

mkdir -p "$flairPathTo"
mkdir -p "$segPathTo"
mkdir -p "$t1PathTo"
mkdir -p "$t1cePathTo"
mkdir -p "$t2PathTo"

for subject in "$1"/*
do
    if [ ! -d "${subject}" ]; then
        echo "$subject"' is not dir.'
        continue
    else
        echo "$subject"' processing...'
    fi

    cp "$subject"/*flair.nii.gz "$flairPathTo"
    cp "$subject"/*seg.nii.gz "$segPathTo"
    cp "$subject"/*t1.nii.gz "$t1PathTo"
    cp "$subject"/*t1ce.nii.gz "$t1cePathTo"
    cp "$subject"/*t2.nii.gz "$t2PathTo"
done
