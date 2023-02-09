mkdir -p data/
cd data/

echo "The raw VIBE estimation of UESTC datasets will be stored in the 'data' folder."


# PHPSDposes
wget --content-disposition https://lsh.paris.inria.fr/surreact/vibe_uestc.tar.gz

tar -xvf vibe_uestc.tar.gz
rm vibe_uestc.tar.gz
