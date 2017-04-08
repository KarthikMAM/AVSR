#preparing for download 
mkdir "data"
cd "data"
mkdir "raw" "audio" "video" "align" "audio/raw" "video/raw" "align/raw"
cd "raw" && mkdir "audio" "video" "align"

for i in `seq $1 $2`
do
    printf "\n\n----------------------------- Downloading Speaker $i -----------------------------\n\n"
    
    if [[ $3 == y ]]
    then
        #download the audio of the ith speaker
        curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/audio/s$i.tar" > "audio/s$i.tar" && printf "\n"
        curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/video/s$i.mpg_vcd.zip" > "video/s$i.zip" && printf "\n"
        curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/align/s$i.tar" > "align/s$i.tar" && printf "\n"
    fi

    if [[ $4 == y ]]
    then
        unzip -q "video/s$i.zip" -d "../video/raw"
        tar -xf "audio/s$i.tar" -C "../audio/raw"
        tar -xf "align/s$i.tar" -C "../align/raw" && mv "../align/raw/align" "../align/raw/s$i"
    fi
done