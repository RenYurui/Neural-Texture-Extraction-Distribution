gdown https://drive.google.com/uc?id=147oB1UvX_mq_2sFNHTyEAUORIROad1VM
gdown https://drive.google.com/uc?id=1CKHtZS8z7hIT6O8AQzSxMN55zGMuLjuJ
mkdir ../result
mkdir ../result/fashion_512
mv epoch_00200_iteration_000495400_checkpoint.pt ../result/fashion_512
mv demo_images.zip ../
cd ..
unzip demo_images.zip 
rm demo_images.zip