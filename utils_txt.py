import re, sys, os
from os.path import basename
import torch
import random
import torchvision.transforms as T
import torchvision
from PIL import Image
import os


def cos_dist(x1, x2):
    return torch.sum(x1 * x2) / (torch.norm(x1) * torch.norm(x2))


def fixed_img_list(lfw_pair_text):
    f = open(lfw_pair_text, 'r')
    lines = []

    while True:
        line = f.readline()
        if not line:
            break
        lines.append(line)
    f.close()

    random.shuffle(lines)
    return lines


def verification(net, pair_list, tst_data_dir, img_size, gray_scale=True):
    similarities = []
    labels = []

    # 이미지 전처리
    t = T.Compose([T.Resize((112, 96)),
                   T.ToTensor(),
                   T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 주어진 모든 이미지 pair에 대해 similarity 계산
    net.eval()
    with torch.no_grad():  # Test 때 GPU를 사용할 경우 메모리 절약을 위해 torch.no_grad() 내에서 하는 것이 좋다.
        for idx, pair in enumerate(pair_list):
            # Read paired images
            path_1, path_2, label = pair.split(' ')
            img_1 = t(Image.open(os.path.join(tst_data_dir, path_1))).unsqueeze(dim=0).cuda()
            img_2 = t(Image.open(os.path.join(tst_data_dir, path_2))).unsqueeze(dim=0).cuda()
            imgs = torch.cat((img_1, img_2), dim=0)

            # Extract feature and save
            features = net.extract_feature(imgs).cpu()
            similarities.append(cos_dist(features[0], features[1]))
            labels.append(int(label))

    '''
    STEP 2 : similarity와 label로 verification accuracy 측정
    '''
    best_accr = 0.0
    best_th = 0.0

    # 각 similarity들이 threshold의 후보가 된다
    list_th = similarities

    # list -> tensor
    similarities = torch.stack(similarities, dim=0)
    labels = torch.ByteTensor(labels)

    # 각 threshold 후보에 대해 best accuracy를 측정
    for i, th in enumerate(list_th):
        pred = (similarities >= th)
        correct = (pred == labels)
        accr = torch.sum(correct).item() / correct.size(0)

        if accr > best_accr:
            best_accr = accr
            best_th = th.item()

    return best_accr, best_th


def make_dir(args):
    model_path = os.path.join(args.exp, args.model_dir)
    log_path = os.path.join(args.exp, args.log_dir)

    if not os.path.isdir(args.exp):
        os.mkdir(args.exp)

    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    if not os.path.isdir(log_path):
        os.mkdir(log_path)


def path2age(path, pat, pos):
    return int(re.split(pat, basename(path))[pos])


def accuracy(preds, labels):
    return (preds.squeeze() == labels.squeeze()).float().mean()


def erase_print(content):
    sys.stdout.write('\033[2K\033[1G')
    sys.stdout.write(content)
    sys.stdout.flush()


def mkdir_p(path):
    try:
        os.makedirs(os.path.abspath(path))
    except OSError as exc:
        if exc.errno == os.errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_ctx(n):
    return torch.device(f'cuda:{n}') if n >= 0 else torch.device('cpu')


def add_grid_img(i, result_img, tst_data_dir, AB_sim, AB, t):
    if i >= len(AB_sim):
        if gray_scale:
            img = torch.randn(1, 1, 128, 128)
        else:
            img = torch.randn(1, 3, 128, 128)

        result_img.append(img)
        result_img.append(img)
    else:
        norm_img = Image.open(os.path.join(tst_data_dir, AB[2 * i]))
        draw = ImageDraw.Draw(norm_img)
        draw.text((45, 15), text=str(round(AB_sim[i].item(), 2)), align="left")

        result_img.append(t(norm_img).unsqueeze(dim=0))
        result_img.append(t(Image.open(os.path.join(tst_data_dir, AB[2 * i + 1]))).unsqueeze(dim=0))
    return result_img


def make_analysis_img(tst_data_dir, img_list_1, img_list_2, labels, best_pred, best_similarities, transform,
                      gray_scale=True):
    TP = []
    TP_sim = []
    FN = []
    FN_sim = []
    FP = []
    FP_sim = []
    TN = []
    TN_sim = []

    for iter_, (i, j, sim) in enumerate(zip(labels, best_pred, best_similarities)):
        if i == j and i == 1:
            TP.append(img_list_1[iter_])
            TP.append(img_list_2[iter_])
            TP_sim.append(best_similarities[iter_])
        elif i == j and i == 0:
            FN.append(img_list_1[iter_])
            FN.append(img_list_2[iter_])
            FN_sim.append(best_similarities[iter_])
        elif i != j and i == 1:
            TN.append(img_list_1[iter_])
            TN.append(img_list_2[iter_])
            TN_sim.append(best_similarities[iter_])
        elif i != j and i == 0:
            FP.append(img_list_1[iter_])
            FP.append(img_list_2[iter_])
            FP_sim.append(best_similarities[iter_])

    iteration = max(len(TP_sim), len(FN_sim), len(TN_sim), len(FP_sim))

    print("TP_sim : %d, FN_sim : %d, TN_sim : %d, FP_sim : %d" % (len(TP_sim), len(FN_sim), len(TN_sim), len(FP_sim)))

    result_img = []

    for i in range(iteration):
        result_img = add_grid_img(i, result_img, tst_data_dir, TP_sim, TP, transform)
        result_img = add_grid_img(i, result_img, tst_data_dir, FN_sim, FN, transform)
        result_img = add_grid_img(i, result_img, tst_data_dir, TN_sim, TN, transform)
        result_img = add_grid_img(i, result_img, tst_data_dir, FP_sim, FP, transform)

    result_img = torch.cat(result_img, dim=0)
    print(result_img.shape)

    grid_img = torchvision.utils.make_grid(result_img, nrow=8, padding=2)

    # denorm -> permute (PIL Image의 형태에 맞게) -> CPU로 전달
    ndarr = (((grid_img / 2) + 0.5) * 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    img.save("test.png")


def make_analysis_folder(tst_data_dir, img_list_1, img_list_2, labels, best_pred, best_similarities, transform,
                         gray_scale=True):
    TP = []
    TP_sim = []
    FN = []
    FN_sim = []
    FP = []
    FP_sim = []
    TN = []
    TN_sim = []

    for iter_, (i, j, sim) in enumerate(zip(labels, best_pred, best_similarities)):
        if i == j and i == 1:
            TP.append(img_list_1[iter_])
            TP.append(img_list_2[iter_])
            TP_sim.append(best_similarities[iter_])
        elif i == j and i == 0:
            FN.append(img_list_1[iter_])
            FN.append(img_list_2[iter_])
            FN_sim.append(best_similarities[iter_])
        elif i != j and i == 1:
            TN.append(img_list_1[iter_])
            TN.append(img_list_2[iter_])
            TN_sim.append(best_similarities[iter_])
        elif i != j and i == 0:
            FP.append(img_list_1[iter_])
            FP.append(img_list_2[iter_])
            FP_sim.append(best_similarities[iter_])

    iteration = max(len(TP_sim), len(FN_sim), len(TN_sim), len(FP_sim))

    print("TP_sim : %d, FN_sim : %d, TN_sim : %d, FP_sim : %d" % (len(TP_sim), len(FN_sim), len(TN_sim), len(FP_sim)))

    if not os.path.isdir("TP"):
        os.mkdir("TP")

    for i in range(len(TP) // 2):
        result_img = []
        img1 = Image.open(TP[2 * i])
        img2 = Image.open(TP[2 * i + 1])

        result_img.append(transform(img1).unsqueeze(dim=0))
        result_img.append(transform(img2).unsqueeze(dim=0))

        result_img2 = torch.cat(result_img, dim=0)

        grid_img = torchvision.utils.make_grid(result_img2, nrow=2, padding=2)

        ndarr = (((grid_img / 2) + 0.5) * 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save("TP/" + TP[2 * i].split("/")[-1] + "_" + str(round(TP_sim[i].item(), 4)) + ".png")

    if not os.path.isdir("FN"):
        os.mkdir("FN")

    for i in range(len(FN) // 2):
        result_img = []
        img1 = Image.open(FN[2 * i])
        img2 = Image.open(FN[2 * i + 1])

        result_img.append(transform(img1).unsqueeze(dim=0))
        result_img.append(transform(img2).unsqueeze(dim=0))

        result_img2 = torch.cat(result_img, dim=0)

        grid_img = torchvision.utils.make_grid(result_img2, nrow=2, padding=2)

        ndarr = (((grid_img / 2) + 0.5) * 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save("FN/" + FN[2 * i].split("/")[-1] + "_" + str(round(FN_sim[i].item(), 4)) + ".png")

    if not os.path.isdir("TN"):
        os.mkdir("TN")

    for i in range(len(TN) // 2):
        result_img = []
        img1 = Image.open(TN[2 * i])
        img2 = Image.open(TN[2 * i + 1])

        result_img.append(transform(img1).unsqueeze(dim=0))
        result_img.append(transform(img2).unsqueeze(dim=0))

        result_img2 = torch.cat(result_img, dim=0)

        grid_img = torchvision.utils.make_grid(result_img2, nrow=2, padding=2)

        ndarr = (((grid_img / 2) + 0.5) * 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save("TN/" + TN[2 * i].split("/")[-1] + "_" + str(round(TN_sim[i].item(), 4)) + ".png")

    if not os.path.isdir("FP"):
        os.mkdir("FP")

    for i in range(len(FP) // 2):
        result_img = []
        img1 = Image.open(FP[2 * i])
        img2 = Image.open(FP[2 * i + 1])

        result_img.append(transform(img1).unsqueeze(dim=0))
        result_img.append(transform(img2).unsqueeze(dim=0))

        result_img2 = torch.cat(result_img, dim=0)

        grid_img = torchvision.utils.make_grid(result_img2, nrow=2, padding=2)

        ndarr = (((grid_img / 2) + 0.5) * 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save("FP/" + FP[2 * i].split("/")[-1] + "_" + str(round(FP_sim[i].item(), 4)) + ".png")


def make_analysis_folder2(tst_data_dir, img_list_1, img_list_2, labels, best_pred, best_similarities, transform,
                          gray_scale=True):
    TP = []
    TP_sim = []
    FN = []
    FN_sim = []
    FP = []
    FP_sim = []
    TN = []
    TN_sim = []

    for iter_, (i, j, sim) in enumerate(zip(labels, best_pred, best_similarities)):
        if i == j and i == 1:
            TP.append(img_list_1[iter_])
            TP.append(img_list_2[iter_])
            TP_sim.append(best_similarities[iter_])
        elif i == j and i == 0:
            FN.append(img_list_1[iter_])
            FN.append(img_list_2[iter_])
            FN_sim.append(best_similarities[iter_])
        elif i != j and i == 1:
            TN.append(img_list_1[iter_])
            TN.append(img_list_2[iter_])
            TN_sim.append(best_similarities[iter_])
        elif i != j and i == 0:
            FP.append(img_list_1[iter_])
            FP.append(img_list_2[iter_])
            FP_sim.append(best_similarities[iter_])

    iteration = max(len(TP_sim), len(FN_sim), len(TN_sim), len(FP_sim))

    print("TP_sim : %d, FN_sim : %d, TN_sim : %d, FP_sim : %d" % (len(TP_sim), len(FN_sim), len(TN_sim), len(FP_sim)))

    if not os.path.isdir("TP"):
        os.mkdir("TP")

    for i in range(len(TP) // 2):
        result_img = []
        img1 = Image.open(TP[2 * i])
        img2 = Image.open(TP[2 * i + 1])

        a = TP[2 * i].split("/")[-2:]
        b = TP[2 * i + 1].split("/")[-2:]

        img3 = Image.open(os.path.join('/home/nas3_userL/Face_dataset/AgeDB_align', "/".join(a)))
        img4 = Image.open(os.path.join('/home/nas3_userL/Face_dataset/AgeDB_align', "/".join(b)))

        result_img.append(transform(img1).unsqueeze(dim=0))
        result_img.append(transform(img2).unsqueeze(dim=0))

        result_img.append(transform(img3).unsqueeze(dim=0))
        result_img.append(transform(img4).unsqueeze(dim=0))

        result_img2 = torch.cat(result_img, dim=0)

        grid_img = torchvision.utils.make_grid(result_img2, nrow=2, padding=2)

        ndarr = (((grid_img / 2) + 0.5) * 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save("TP/" + TP[2 * i].split("/")[-1] + "_" + str(round(TP_sim[i].item(), 4)) + ".png")

    if not os.path.isdir("FN"):
        os.mkdir("FN")

    for i in range(len(FN) // 2):
        result_img = []
        img1 = Image.open(FN[2 * i])
        img2 = Image.open(FN[2 * i + 1])

        a = FN[2 * i].split("/")[-2:]
        b = FN[2 * i + 1].split("/")[-2:]

        img3 = Image.open(os.path.join('/home/nas3_userL/Face_dataset/AgeDB_align', "/".join(a)))
        img4 = Image.open(os.path.join('/home/nas3_userL/Face_dataset/AgeDB_align', "/".join(b)))

        result_img.append(transform(img1).unsqueeze(dim=0))
        result_img.append(transform(img2).unsqueeze(dim=0))

        result_img.append(transform(img3).unsqueeze(dim=0))
        result_img.append(transform(img4).unsqueeze(dim=0))

        result_img2 = torch.cat(result_img, dim=0)

        grid_img = torchvision.utils.make_grid(result_img2, nrow=2, padding=2)

        ndarr = (((grid_img / 2) + 0.5) * 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save("FN/" + FN[2 * i].split("/")[-1] + "_" + str(round(FN_sim[i].item(), 4)) + ".png")

    if not os.path.isdir("TN"):
        os.mkdir("TN")

    for i in range(len(TN) // 2):
        result_img = []
        img1 = Image.open(TN[2 * i])
        img2 = Image.open(TN[2 * i + 1])

        a = TN[2 * i].split("/")[-2:]
        b = TN[2 * i + 1].split("/")[-2:]

        img3 = Image.open(os.path.join('/home/nas3_userL/Face_dataset/AgeDB_align', "/".join(a)))
        img4 = Image.open(os.path.join('/home/nas3_userL/Face_dataset/AgeDB_align', "/".join(b)))

        result_img.append(transform(img1).unsqueeze(dim=0))
        result_img.append(transform(img2).unsqueeze(dim=0))

        result_img.append(transform(img3).unsqueeze(dim=0))
        result_img.append(transform(img4).unsqueeze(dim=0))

        result_img2 = torch.cat(result_img, dim=0)

        grid_img = torchvision.utils.make_grid(result_img2, nrow=2, padding=2)

        ndarr = (((grid_img / 2) + 0.5) * 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save("TN/" + TN[2 * i].split("/")[-1] + "_" + str(round(TN_sim[i].item(), 4)) + ".png")

    if not os.path.isdir("FP"):
        os.mkdir("FP")

    for i in range(len(FP) // 2):
        result_img = []
        img1 = Image.open(FP[2 * i])
        img2 = Image.open(FP[2 * i + 1])

        a = FP[2 * i].split("/")[-2:]
        b = FP[2 * i + 1].split("/")[-2:]

        img3 = Image.open(os.path.join('/home/nas3_userL/Face_dataset/AgeDB_align', "/".join(a)))
        img4 = Image.open(os.path.join('/home/nas3_userL/Face_dataset/AgeDB_align', "/".join(b)))

        result_img.append(transform(img1).unsqueeze(dim=0))
        result_img.append(transform(img2).unsqueeze(dim=0))

        result_img.append(transform(img3).unsqueeze(dim=0))
        result_img.append(transform(img4).unsqueeze(dim=0))

        result_img2 = torch.cat(result_img, dim=0)

        grid_img = torchvision.utils.make_grid(result_img2, nrow=2, padding=2)

        ndarr = (((grid_img / 2) + 0.5) * 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save("FP/" + FP[2 * i].split("/")[-1] + "_" + str(round(FP_sim[i].item(), 4)) + ".png")


def make_for_lifespan(tst_data_dir, img_list_1, img_list_2, labels, best_pred, best_similarities, transform,
                      gray_scale=True):
    TP = []
    TP_sim = []
    FN = []
    FN_sim = []
    FP = []
    FP_sim = []
    TN = []
    TN_sim = []

    for iter_, (i, j, sim) in enumerate(zip(labels, best_pred, best_similarities)):
        if i == j and i == 1:
            TP.append(img_list_1[iter_])
            TP.append(img_list_2[iter_])
            TP_sim.append(best_similarities[iter_])
        elif i == j and i == 0:
            FN.append(img_list_1[iter_])
            FN.append(img_list_2[iter_])
            FN_sim.append(best_similarities[iter_])
        elif i != j and i == 1:
            TN.append(img_list_1[iter_])
            TN.append(img_list_2[iter_])
            TN_sim.append(best_similarities[iter_])
        elif i != j and i == 0:
            FP.append(img_list_1[iter_])
            FP.append(img_list_2[iter_])
            FP_sim.append(best_similarities[iter_])

    iteration = max(len(TP_sim), len(FN_sim), len(TN_sim), len(FP_sim))

    print("TP_sim : %d, FN_sim : %d, TN_sim : %d, FP_sim : %d" % (len(TP_sim), len(FN_sim), len(TN_sim), len(FP_sim)))

    if not os.path.isdir("TN_life"):
        os.mkdir("TN_life")
        os.mkdir("TN_life/A")
        os.mkdir("TN_life/B")

    for i in range(len(TN) // 2):
        result_img = []
        img1 = Image.open(os.path.join(tst_data_dir, TN[2 * i]))
        img2 = Image.open(os.path.join(tst_data_dir, TN[2 * i + 1]))

        img1.save("TN_life/A/" + TN[2 * i].split("/")[-1] + "_" + str(round(TN_sim[i].item(), 4)) + ".png")
        img2.save("TN_life/B/" + TN[2 * i + 1].split("/")[-1] + "_" + str(round(TN_sim[i].item(), 4)) + ".png")

    if not os.path.isdir("FP_life"):
        os.mkdir("FP_life")
        os.mkdir("FP_life/A")
        os.mkdir("FP_life/B")

    for i in range(len(FP) // 2):
        result_img = []
        img1 = Image.open(os.path.join(tst_data_dir, FP[2 * i]))
        img2 = Image.open(os.path.join(tst_data_dir, FP[2 * i + 1]))

        img1.save("FP_life/A/" + FP[2 * i].split("/")[-1] + "_" + str(round(FP_sim[i].item(), 4)) + ".png")
        img2.save("FP_life/B/" + FP[2 * i + 1].split("/")[-1] + "_" + str(round(FP_sim[i].item(), 4)) + ".png")


def make_false_txt(tst_data_dir, img_list_1, img_list_2, labels, best_pred, best_similarities, transform,
                   gray_scale=True):
    TP = []
    TP_sim = []
    FN = []
    FN_sim = []
    FP = []
    FP_sim = []
    TN = []
    TN_sim = []

    for iter_, (i, j, sim) in enumerate(zip(labels, best_pred, best_similarities)):
        if i == j and i == 1:
            TP.append(img_list_1[iter_])
            TP.append(img_list_2[iter_])
            TP_sim.append(best_similarities[iter_])
        elif i == j and i == 0:
            FN.append(img_list_1[iter_])
            FN.append(img_list_2[iter_])
            FN_sim.append(best_similarities[iter_])
        elif i != j and i == 1:
            TN.append(img_list_1[iter_])
            TN.append(img_list_2[iter_])
            TN_sim.append(best_similarities[iter_])
        elif i != j and i == 0:
            FP.append(img_list_1[iter_])
            FP.append(img_list_2[iter_])
            FP_sim.append(best_similarities[iter_])

    iteration = max(len(TP_sim), len(FN_sim), len(TN_sim), len(FP_sim))

    print("TP_sim : %d, FN_sim : %d, TN_sim : %d, FP_sim : %d" % (len(TP_sim), len(FN_sim), len(TN_sim), len(FP_sim)))

    file = open("false.txt", "w")

    for i in range(len(TN) // 2):
        data = "%s %s %s 1\n" % (TN[2 * i], TN[2 * i + 1], str(round(TN_sim[i].item(), 4)))
        file.write(data)

    for i in range(len(FP) // 2):
        data = "%s %s %s 0\n" % (FP[2 * i], FP[2 * i + 1], str(round(FP_sim[i].item(), 4)))
        file.write(data)

    file.close()

