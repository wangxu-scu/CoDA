cd ../
# dataset='office31'; class_num=31; cluster_num=50; tasks='w2d' #a2w d2a d2w w2a w2d'
# dataset='image_CLEF'; class_num=12; cluster_num=12; tasks='b2c b2i b2p c2b c2i c2p i2b i2c i2p p2b p2c p2i'
dataset='office_home'; class_num=65; cluster_num=100; tasks='a2c' #a2c a2p p2r #a2c a2p a2r c2p c2r p2r' #c2r c2p p2r' #'a2c a2p a2r c2a c2p c2r p2a p2c p2r r2a r2c r2p'
# dataset='adaptiope'; class_num=123; cluster_num=100; tasks='r2s' #p2s r2p r2s s2p s2r'


gpu_id=0

in_domain=True
cross_domain=True
kmeans_all_features=True
lambda_cross_domain=0.0
save_prefix='multi_cluster'

for t in ${tasks};
do
    python -u train_udar_oh_coda.py --dset ${t} --dataset ${dataset} --class_num ${class_num} --cluster_num ${cluster_num} --gpu_id ${gpu_id} \
    --nce-m 0.95 --in_domain ${in_domain} --cross_domain ${cross_domain} --save_prefix ${save_prefix} --kmeans_all_features ${kmeans_all_features} \
    --cross_domain_loss 'l1' --lambda_cross_domain ${lambda_cross_domain} --cross_domain_softmax False --seed 0
done
