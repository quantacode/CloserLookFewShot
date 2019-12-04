

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

## Adaptation by Finetuning ###
## complete target domain base split
#python ./basic_train.py --train_aug \
#--dataset emnist \
#--model Conv4S \
#--method Standard \
#--stop_epoch 10000 \
#--lr 0.001 \
#--load_modelpth '/home/rajshekd/projects/FSG/PRODA/checkpoints/cross_char/Conv4S_protonet_5way_5shot/vanila-Protonet/best_model.tar' \
#--exp_id 'adaptFT-allNovelTrainClasses_lr-0.001'

## few-shot
#python ./train.py \
#--dataset cross_char \
#--model Conv4S \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--n_shot 1 \
#--adaptFinetune \
#--exp_id 'adaptFT_wt-0.1_lr-0.001'


#################### miniImageNet-CUB ##############
## no-adapt
#python ./train.py --train_aug \
#--dataset cross \
#--model ResNet34 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--exp_id 'vanila'
##--load_modelpth '/home/rajshekd/projects/FSG/CloserLookFewShot/checkpoints/pretrained-imagenet/model.tar' \

##adversarial
#python ./train.py --train_aug \
#--dataset cross \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.00001 \
#--adversarial \
#--gamma 1.0 \
#--load_modelpth '/home/rajshekd/projects/FSG/PRODA/checkpoints/cross/ResNet18_protonet_aug_5way_5shot/vanila_pretrain-Imagenet/best_model.tar' \
#--exp_id 'adversarial-ConcatZ_domainReg-1.0_lr-0.00001_DiscM-4096_base2base_SPL'

### Adaptation by Finetuning ###
## few-shot
#python ./train.py --train_aug \
#--dataset cross \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--adaptFinetune \
#--exp_id 'adaptFT_wt-0.1_lr-0.001'

## complete target domain base split
#python ./basic_train.py --train_aug \
#--dataset CUB \
#--model ResNet18 \
#--method Standard \
#--stop_epoch 10000 \
#--lr 0.001 \
#--load_modelpth '/home/rajshekd/projects/FSG/PRODA/checkpoints/cross/ResNet18_protonet_aug_5way_5shot/vanila_pretrain-Imagenet/best_model.tar' \
#--exp_id 'adaptFT-allNovelTrainClasses_lr-0.001'


############### miniImageNet-flowers ##############
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

###################### VGG_flowers, CUB ##############
## no-adapt
#python ./train.py --train_aug \
#--dataset flowers_CUB \
#--model ResNet34 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.001 \
#--exp_id 'vanila'

### adversarial
#python ./train.py --train_aug \
#--dataset flowers_CUB \
#--model ResNet34 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.0001 \
#--gamma 1.0 \
#--adversarial \
#--load_modelpth '/home/rajshekd/projects/FSG/PRODA/checkpoints/flowers_CUB/ResNet34_protonet_aug_5way_5shot/vanila/best_model.tar' \
#--exp_id 'adversarial-ConcatZ_domainReg-0.1_lr-0.0001_DiscM-4096_Base2Base_SPL'

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

################################## ADDA #############################################
#python ./train_adda.py --train_aug \
#--dataset cross \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.00001 \
#--load_modelpth '/home/rajshekd/projects/FSG/PRODA/checkpoints/cross/ResNet18_protonet_aug_5way_5shot/vanila_pretrain-Imagenet/best_model.tar' \
#--exp_id 'ADDA_disc-4096'

#python ./train_adda.py \
#--dataset cross_char \
#--model Conv4 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.00001 \
#--n_shot 1 \
#--load_modelpth '/home/rajshekd/projects/FSG/PRODA/checkpoints/cross_char/Conv4_protonet_5way_1shot/vanila/best_model.tar' \
#--exp_id 'ADDA_disc-512'

################################## DAN #############################################
#python ./train_dan.py --train_aug \
#--dataset cross \
#--model ResNet18 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.0001 \
#--dan --gamma 1.0 \
#--load_modelpth '/home/rajshekd/projects/FSG/PRODA/checkpoints/cross/ResNet18_protonet_aug_5way_5shot/vanila_pretrain-Imagenet/best_model.tar' \
#--exp_id 'DAN_advLossWt-1.0'

#python ./train_dan.py \
#--dataset cross_char \
#--model Conv4 \
#--method protonet \
#--stop_epoch 10000 \
#--lr 0.00001 \
#--n_shot 1 \
#--dan --gamma 1.0 \
#--load_modelpth '/home/rajshekd/projects/FSG/PRODA/checkpoints/cross_char/Conv4_protonet_5way_1shot/vanila/best_model.tar' \
#--exp_id 'DAN_advLossWt-1.0'

############################### EVALUATE #############################################
#python save_features.py --train_aug --dataset CUB_flowers --model ResNet10 --method protonet --n_shot 5 --split all \
#--exp_id 'vanila'

#python test.py --train_aug --dataset flowers_CUB --model ResNet10 --method protonet --n_shot 5 --split novel \
#--exp_id 'adversarial-ConcatZ_domainReg-0.1_lr-0.0001_DiscM-4096_Base2Base_SPL'

#################################### VISUALIZE #############################################
#python visualize_domains.py --dataset cross_char --model Conv4S --method protonet --n_shot 5 --split novel

python evaluate_class_perf.py --train_aug --dataset CUB_flowers --model ResNet10 --method protonet --n_shot 5 --split all \
--exp_id 'adversarial-ConcatZ_domainReg-0.1_lr-0.0001_DiscM-4096_Base2Base'
