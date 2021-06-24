import io
import torch
from torchvision import transforms as trans
from datetime import datetime
from PIL import Image
from data.data_pipe import de_preprocess
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info # REMOVE AT PUBLISH

plt.switch_backend('agg')


def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn



def hflip_batch(imgs_tensor):
    hflip = trans.Compose([
            de_preprocess,
            trans.ToPILImage(),
            trans.functional.hflip,
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs

def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf

# lfw_accuracy, lfw_thres, roc_curve_tensor2, lfw_dist = self.evaluate(conf, self.lfw, self.lfw_issame)
# # NEGATIVE WRONG
# wrong_list = np.where((self.lfw_issame == False) & (lfw_dist < lfw_thres))[0]
# lfw_negative = len(wrong_list)
# # POSITIVE WRONG
# wrong_list = np.where((self.lfw_issame == True) & (lfw_dist > lfw_thres))[0]
# lfw_positive = len(wrong_list)
#
# # FGNETC evaluation
# fgnetc_accuracy, fgnetc_thres, roc_curve_tensor2, fgnetc_dist = self.evaluate(conf, self.fgnetc, self.fgnetc_issame)
# # NEGATIVE WRONG
# wrong_list = np.where((self.fgnetc_issame == False) & (fgnetc_dist < fgnetc_thres))[0]
# fgnetc_negative = len(wrong_list)
# # POSITIVE WRONG
# wrong_list = np.where((self.fgnetc_issame == True) & (fgnetc_dist > fgnetc_thres))[0]
# fgnetc_positive = len(wrong_list)
# # self.board_val('fgent_c', accuracy2, best_threshold2, roc_curve_tensor2, fgnet_negative\, positive_wrong2)
# print(f'fgnetc_acc: {fgnetc_accuracy}')
#
# if self.conf.wandb:
#     wandb.log({
#         "lfw_acc": lfw_accuracy,
#         "lfw_best_threshold": lfw_thres,
#         "lfw_negative_wrong": lfw_negative,
#         "lfw_positive_wrong": lfw_positive,
#
#         "fgnet_c_acc": fgnetc_accuracy,
#         "fgnet_c_best_threshold": fgnetc_thres,
#         "fgnet_c_negative_wrong": fgnetc_negative,
#         "fgnet_c_positive_wrong": fgnetc_positive,
#     }, step=self.step)

# if self.step % self.save_every == 0 and self.step != 0:
#     print('saving model....')
#     # save with most recently calculated accuracy?
#     # if conf.finetune_model_path is not None:
#     #     self.save_state(conf, accuracy2,
#     #                     extra=str(conf.data_mode) + '_' + str(conf.net_depth) + '_' + str(
#     #                         conf.batch_size) + 'finetune')
#     # else:
#     #     self.save_state(conf, accuracy2,extra=str(conf.data_mode) + '_' + str(conf.exp) + '_' + str(conf.batch_size))
#     if self.conf.loss == 'Broad':
#         if lfw_accuracy > best_accuracy:
#             best_accuracy = lfw_accuracy
#             print('saving best model....')
#             self.save_best_state(conf, best_accuracy, extra=str(conf.data_mode) + '_' + str(conf.exp))
#             if lfw_accuracy > 0.99:
#                 import sys
#                 sys.exit(0)
#
#     else:
#         if fgnetc_accuracy > best_accuracy:
#             best_accuracy = fgnetc_accuracy
#             print('saving best model....')
#             self.save_best_state(conf, best_accuracy, extra=str(conf.data_mode) + '_' + str(conf.exp))

def model_profile(net):
    macs, params = get_model_complexity_info(net, (3, 112, 112), as_strings=False, print_per_layer_stat=False)

    print('{:<30}  {:<8}'.format('Computational complexity: ', int(macs)))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))