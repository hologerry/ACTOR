mkdir -p data/
cd data/

echo "The PHPSDposes datasets will be stored in the 'data' folder, it is used to estimate the HumanAct12Poses."


# PHPSDposes
gdown "https://drive.google.com/uc?id=1ErCh_WYKHS2RkoL3A0xb_8TY6rBLJO4W"
unzip pose.zip
mv pose PHPSDposes
rm pose.zip

mkdir phspdCameras
cd ./phspdCameras
gdown "https://drive.google.com/uc?id=1BG4tZxyih1Kv3-slm_f3rbAhsc4hDrTx"
gdown "https://drive.google.com/uc?id=1gsu1tpKQgJOW9eu967QcGjS1qj4407Pm"

cd ../
gdown "https://drive.google.com/uc?id=1yATgxkUbIJlJe8LzLCX42-fwrVv42mvg"
unzip HumanAct12.zip
rm HumanAct12.zip


