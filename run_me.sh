#!/usr/bin/env bash
set -x
source ~/torch/bin/activate

################################# Within Dataset #############################################
####### omniglot ##########
#python ./train.py \
#--dataset omniglot \
#--model Conv4 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--n_shot 5 --n_shot_test 2 \
#--exp_id 'vanila_5to2-shot'

####### miniImageNet, CUB ##########
#python ./train.py \
#--dataset CUB --train_aug \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--n_shot 10 --n_shot_test 5 \
#--exp_id 'vanila_10to5-shot'

## adversarial adaptation (shot analysis )
#python ./train.py \
#--dataset CUB --train_aug \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--adversarial \
#--n_shot 5 --n_shot_test 1 \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/CUB/ResNet18_protonet_aug_5way_5shot/vanila_5to1-shot/best_model.tar' \
#--exp_id 'adversarial_5to1-shot'


######### VGG_flowers ##########
#python ./train.py --train_aug \
#--dataset flowers \
#--model Conv4 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--exp_id 'vanila_endEpoch-10000'

######### OfficeHome Product ##########
#python ./train.py --train_aug \
#--dataset officeProduct \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/cross/ResNet18_protonet_aug_5way_5shot/vanila_endEpoch-10000/best_model.tar' \
#--exp_id 'vanila_pretrain-ResNet18'


################################ cross-domain #############################################

############### Omniglot-EMNIST ####################
## no-adapt
#python ./train.py \
#--dataset cross_char \
#--model Conv4 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--exp_id 'vanila'

## adversarial
#python ./train.py \
#--dataset cross_char \
#--model Conv4 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--adversarial \
#--exp_id 'adversarial-ConcatZ_domainReg-0.1_lr-0.001_endEpoch-10000_DiscM-2FC512_run2'

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

################ miniImageNet-CUB ##############
## no-adapt
#python ./train.py --train_aug \
#--dataset cross \
#--model ResNet18_plus \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--n_shot 20 \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/miniImagenet/ResNet18_baseline_aug/vanila/best_model.tar' \
#--exp_id 'vanila_pretrain-miniImagenet_allCat_512|1024'

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
#--exp_id 'adversarial-ConcatZ_domainReg-0.1_lr-0.001_endEpoch-10000_DiscM-4096_warmStart'

############# miniImageNet-flowers ##############
## no-adapt
#python ./train.py --train_aug \
#--dataset miniImagenet_flowers \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--exp_id 'vanila_endEpoch-10000'

## adversarial
#python ./train.py --train_aug \
#--dataset miniImagenet_flowers \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--adversarial \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/miniImagenet_flowers/ResNet18_protonet_aug_5way_5shot/vanila_endEpoch-10000_shots-5/300.tar' \
#--exp_id 'adversarial-ConcatZ_domainReg-0.1_lr-0.001_endEpoch-10000_DiscM-4096_BaseNovel'

#

################# VGG_flowers, CUB ##############
## no-adapt
#python ./train.py --train_aug \
#--dataset CUB_flowers \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--exp_id 'vanila_endEpoch-10000'

### adversarial
#python ./train.py --train_aug \
#--dataset flowers_CUB \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--adversarial \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/flowers_CUB/ResNet18_protonet_aug_5way_5shot/vanila_endEpoch-10000/snap/best_model.tar' \
#--exp_id 'adversarial-SeparateZ_domainReg-0.1_lr-0.001_endEpoch-10000_DiscM-4096'

############### Office Home ##############
# no-adapt
python ./train.py --train_aug \
--dataset product_clipart \
--model ResNet18_plus \
--method protonet \
--stop_epoch 10000 \
--lr 0.001 \
--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/miniImagenet/ResNet18_baseline_aug/vanila/best_model.tar' \
--exp_id 'vanila_pretrain-miniImagenet-allCat_512|1024'

### adversarial
#python ./train.py --train_aug \
#--dataset c \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--adversarial \
#--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/product_clipart/ResNet18_protonet_aug_5way_5shot/vanila_pretrain-ImageNet/best_model.tar' \
#--exp_id 'adversarial-ConcatZ_domainReg-0.1_lr-0.001_DiscM-4096_BaseNovel'


################################# EVALUATE #############################################
#python save_features.py --dataset cross_char --model Conv4 --method protonet \
#--exp_id 'adversarial-separateZ_domainReg-0.1_lr-0.0001_endEpoch-10000_DiscM-2FC512'
#
#python test.py --dataset cross_char --model Conv4 --method protonet \
#--exp_id 'adversarial-separateZ_domainReg-0.1_lr-0.0001_endEpoch-10000_DiscM-2FC512'
