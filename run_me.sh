

#!/usr/bin/env bash
set -x
source ~/torch/bin/activate

################################## Prior Works #############################################
#python ./train.py --train_aug \
#--dataset cross \
#--model Conv6 \
#--method maml \
#--lr 0.001 \
#--n_shot 5 \
#--exp_id 'vanila'

################################# Within Dataset #############################################
############## omniglot ##########
#python ./train.py \
#--dataset omniglot \
#--model Conv4 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--exp_id 'vanila'

### shot analysis
#python ./train.py \
#--dataset omniglot \
#--model Conv4 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--n_shot 10 --n_shot_test 20 \
#--exp_id 'vanila_10to20-shot'

####### miniImageNet, CUB ##########
#python ./train.py \
#--dataset miniImagenet --train_aug \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--n_shot 20 \
#--exp_id 'vanila'

## shot analysis
#python ./train.py \
#--dataset CUB --train_aug \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--adversarial \
#--n_shot 10 --n_shot_test 5 \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/CUB/ResNet18_protonet_aug_5way_10shot/vanila_10to2-shot/best_model.tar' \
#--exp_id 'adversarial_10to2-shot_DiscM-4096'


######### VGG_flowers ##########
#python ./train.py --train_aug \
#--dataset flowers \
#--model Conv4 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--exp_id 'vanila_endEpoch-10000'

### shot analysis
#python ./train.py --train_aug \
#--dataset flowers \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--n_shot 10 --n_shot_test 2 \
#--lr 0.001 \
#--exp_id 'vanila_shot-10to2'


######### OfficeHome Product ##########
#python ./train.py --train_aug \
#--dataset officeProduct \
#--model ResNet10 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/miniImagenet/ResNet18_protonet_2way_20shot/vanila_endEpoch-10000_shots-20/best_model.tar' \
#--exp_id 'vanila_pretrain-miniImageNet'


################################ cross-domain #############################################

################ Omniglot-EMNIST ####################
## no-adapt
#python ./train.py \
#--dataset cross_char \
#--model Conv4 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--n_shot 1 \
#--exp_id 'vanila'

### adversarial
#python ./train.py \
#--dataset cross_char \
#--model Conv4S \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.0001 \
#--adversarial \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/cross_char/Conv4S_protonet_5way_5shot/vanila_endEpoch-4000/best_model.tar' \
#--exp_id 'temp'

## adversarial
#python ./train.py --train_aug \
#--dataset cross \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--n_shot 20 \
#--adversarial \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/cross/ResNet18_protonet_aug_5way_20shot/vanila_endEpoch-10000_shots-20/snap/best_model.tar' \
#--exp_id 'adversarial-ConcatZ_domainReg-1.0_lr-0.001_endEpoch-10000_DiscM-512|512_warmStart'

#################### miniImageNet-CUB ##############
## no-adapt
#python ./train.py --train_aug \
#--dataset cross \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--n_shot 1 \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/pretrained-imagenet/model.tar' \
#--exp_id 'vanila'

##adversarial
#python ./train.py --train_aug \
#--dataset cross \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.0001 \
#--adversarial \
#--n_shot 1 \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/cross/ResNet18_protonet_aug_5way_1shot/vanila/500.tar' \
#--exp_id 'adversarial-ConcatZ_domainReg-1.0_lr-0.0001_DiscM-4096_SPL-curriculum-50_base2base'

############## miniImageNet-flowers ##############
## no-adapt
#python ./train.py --train_aug \
#--dataset miniImagenet_flowers \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--n_shot 1 \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/pretrained-imagenet/model.tar' \
#--exp_id 'vanila'

## adversarial
#python ./train.py --train_aug \
#--dataset miniImagenet_flowers \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.0001 \
#--adversarial \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/miniImagenet_flowers/ResNet18_protonet_aug_5way_5shot/vanila_endEpoch-10000_shots-5/best_model.tar' \
#--exp_id 'adversarial-ConcatZ_domainReg-1.0_lr-0.0001_endEpoch-10000_DiscM-4096_tanh-curriculum-scale200'

##################### VGG_flowers, CUB ##############
## no-adapt
#python ./train.py --train_aug \
#--dataset CUB_flowers \
#--model Conv4 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--exp_id 'vanila_base2base_10to1'

## adversarial
python ./train.py --train_aug \
--dataset CUB_flowers \
--model ResNet10 \
--method protonet \
--stop_epoch 10000 \
--lr 0.0001 \
--adversarial \
--n_shot 10 --n_shot_test 1 \
--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/CUB_flowers/ResNet10_protonet_aug_5way_10shot/vanila_shot-10to10/best_model.tar' \
--exp_id 'adversarial-ConcatZ_domainReg-1.0_lr-0.0001_endEpoch-10000_DiscM-4096_Base2Base_SPL'

################# Office Home ##############
## no-adapt
#python ./train.py --train_aug \
#--dataset product_clipart \
#--model Conv4 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--exp_id 'vanila'

### adversarial
#python ./train.py --train_aug \
#--dataset product_clipart \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.0001 \
#--adversarial \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/product_clipart/ResNet18_protonet_aug_5way_5shot/vanila_pretrain-Imagenet/best_model.tar' \
#--exp_id 'adversarial-ConcatZ_domainReg-1.0_lr-0.0001_DiscM-4096_SPL'


##
################################# EVALUATE #############################################
#python save_features.py  --dataset cross_char --model Conv4 --method protonet --n_shot 1 --split novel \
#--exp_id 'adversarial-ConcatZ_domainReg-0.1_lr-0.0001_endEpoch-10000_DiscM-4096_SPL_base2novel'
#
#python test.py  --dataset cross_char --model Conv4 --method protonet --n_shot 1 --split novel \
#--exp_id 'adversarial-ConcatZ_domainReg-0.1_lr-0.0001_endEpoch-10000_DiscM-4096_SPL_base2novel'
