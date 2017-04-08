ROOT_DIR=$PWD
cd ./data/video/raw

print "---------------------------------------PROCESSING VIDEO:START---------------------------------------"

for speaker in *
do
    cd $speaker

    for video in *.mpg
    do
        target=../../frames/$speaker/${video%.*}
        mkdir -p -m 777 "$target"
        ffmpeg -loglevel panic -i $video "$target/frame_%04d.jpg"

        python -W ignore "$ROOT_DIR/video_mouth_extraction.py" $target
    done

    cd ..
done

cd ../../../

print "--------------------------------------PROCESSING VIDEO:SUCCESS--------------------------------------"