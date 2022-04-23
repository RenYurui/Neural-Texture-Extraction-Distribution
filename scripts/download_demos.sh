gdown https://drive.google.com/uc?id=147oB1UvX_mq_2sFNHTyEAUORIROad1VM
gdown https://drive.google.com/uc?id=1eM2ikE2o0T5376rAV5nrTNjDE4Rh18_a
mkdir ../result
mkdir ../result/fashion_512
mkdir ../result/fashion_256
mv epoch_00200_iteration_000495400_checkpoint.pt ../result/fashion_512
mv demo_images.zip ../
gdown https://drive.google.com/uc?id=1CnXLtpTGSKHMeOyyjd5GkaMVIF2eBtkz
mv epoch_00200_iteration_000495400_checkpoint.pt ../result/fashion_256
cd ..
unzip demo_images.zip 
rm demo_images.zip