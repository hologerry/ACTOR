mkdir -p models
cd models/

echo -e "The smpl files will be stored in the 'models/smpl/' folder"
gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"
rm -rf smpl

unzip smpl.zip
echo -e "Cleaning"
rm smpl.zip

echo -e "Downloading done!"
